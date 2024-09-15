# -*- coding: utf-8 -*-
"""
Script Name: Sentiment Analysis with PySpark and State-of-the-Art Transformers
Description: This script leverages PySpark, Spark NLP, and Hugging Face Transformers to perform
             sentiment analysis on large-scale textual data. It is designed to efficiently
             process data in a distributed environment using robust NLP pipelines and machine learning
             techniques. The script supports multiple sentiment analysis models, enables text
             preprocessing into sentences, and computes sentiment scores using both pre-trained
             and fine-tuned transformer models.

Author: Anmol Jaiswal
Created on: 17 Jun' 2024
Last updated: 15 July' 2024
Version: 1.1
"""

####################################################################################
##                            Importing Modules & Libraries                       ##
####################################################################################
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import time
import torch
torch.cuda.empty_cache()

from loguru import logger

import math
import numpy as np
import pandas as pd
from functools import reduce
from joblib import Parallel, delayed
from itertools import chain
import uuid
import shutil

from torch.nn import DataParallel
import torch.distributed as dist
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
# from optimum.onnxruntime import ORTModelForSequenceClassification

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import DoubleType, FloatType, ArrayType

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import SentenceDetector

import mlflow
import mlflow.spark
experiment_name = "/Users/anmol@i-genie.ai/SentimentAnalysisGPU/Sentiment_Analyzer_Framework_Logs"
mlflow.set_experiment(experiment_name)

####################################################################################
##                               Configuration Setup                              ##
####################################################################################
MODEL_DIRECTORY = "/dbfs/mnt/igenie-blob01/Anmol_AI_dir/model_dir"
BATCH_SIZE = 128
MAX_LENGTH = [60, 16]
CHUNK_BASE = 200000

####################################################################################
##                           Sentiment Analyzer Class                             ##
####################################################################################

class SentimentAnalyzer:
    def __init__(self, spark_session, transformer_batch_size=BATCH_SIZE, device=None):
        """
        Initialize the SentimentAnalyzer class.

        :param spark_session: Spark session object
        :param device: 'cpu' or 'gpu' to specify device for transformer models
        :param transformer_batch_size: Batch size for transformer model inference
        """
        assert transformer_batch_size > 0, "Batch size must be positive"
        self.spark = spark_session
        self.spark.sparkContext.setCheckpointDir("dbfs:/tmp/sentiment_analysis_log/")

        self.transformer_batch_size = transformer_batch_size
        self.device = self._get_device(device)
        self._setup_transformer_pipeline()
        logger.success(f"SentimentAnalyzer instance created - {self.device}")
    
    def _get_device(self, device):
        """
        Determine the device to use (CPU or GPU).

        :param device: 'cpu', 'gpu', or None to automatically detect
        :return: Device identifier for transformers pipeline
        """
        if device:
            return 0 if device == 'gpu' else -1
        return 0 if torch.cuda.is_available() else -1
    
    def _setup_sentparser_pipeline(self, text_column: str):
        """
        Setup the Spark NLP pipeline for sentence parsing.

        :param text_column: The name of the column containing text to be parsed
        """
        document_assembler = DocumentAssembler() \
            .setInputCol(text_column) \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentences")

        finisher = Finisher() \
            .setInputCols(["sentences"]) \
            .setOutputCols(["sentences_array"]) \
            .setOutputAsArray(True)
        
        self.sentParser_pipeline = Pipeline(stages=[
            document_assembler, 
            sentence_detector, 
            finisher
        ])
    
    def _setup_transformer_pipeline(self):
        """
        Setup the transformer model and tokenizer for sentiment analysis.
        """
        try:
            save_directory = MODEL_DIRECTORY
            model_save_path = f"{save_directory}/roberta-model"
            tokenizer_save_path = f"{save_directory}/roberta-tokenizer"
            
            # ort_model = ORTModelForSequenceClassification.from_pretrained(
            #             model_save_path,
            #             export=True,
            #             provider="CUDAExecutionProvider",
            #             )
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_save_path, quantization_config=quantization_config, low_cpu_mem_usage=True)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
            self.model.eval()
            logger.success("Transformer model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to setup transformer model and tokenizer: {e}")
            raise e
    
    def repartition_based_on_sentences(self, df: DataFrame, text_column: str):
        """
        Repartition the DataFrame based on the total number of sentences after sentence detection.

        :param df: Spark DataFrame containing the data
        :param initial_partitions: Initial number of partitions
        :param text_column: The name of the column containing text
        :return: Repartitioned DataFrame
        """
        # Apply the sentence detection pipeline
        self._setup_sentparser_pipeline(text_column)
        sent_parser_model = self.sentParser_pipeline.fit(df)
        sent_parsed_result = sent_parser_model.transform(df)

        # # Calculate the total number of sentences
        # sentence_counts = sent_parsed_result.withColumn("sentence_count", F.size(F.col("sentences_array")))
        # total_sentences = sentence_counts.agg(F.sum("sentence_count")).collect()[0][0]
        
        # Calculate the new number of partitions
        initial_partitions = df.rdd.getNumPartitions()
        initial_rows = df.count()
        # new_partitions = 2 + int(initial_partitions * math.ceil(total_sentences // initial_rows))
        new_partitions = 2 + int(math.ceil(initial_rows // 100000))
        
        # Repartition the DataFrame
        if (new_partitions > initial_partitions) or (2*new_partitions < initial_partitions):
            repartitioned_df = sent_parsed_result.repartition(new_partitions)
            repartitioned_df = repartitioned_df.checkpoint()
            torch.cuda.empty_cache()
            logger.success(f"Repartitioned : From {initial_partitions} to {repartitioned_df.rdd.getNumPartitions()}")
            return repartitioned_df
        else:
            # repartitioned_df = sent_parsed_result.checkpoint()
            return sent_parsed_result
    
    def trigger_SentimentInference(self, df: DataFrame, text_column="text", sentParse=True, binarize=False, enable_batching='dynamic'):
        """
        Trigger the sentiment inference on the given DataFrame.

        :param df: Spark DataFrame containing the data
        :param text_column: The name of the column containing text
        :param sentParse: Whether to parse sentences before inference
        :param binarize: Whether to convert scores to binary values
        :param enable_batching: ['dynamic', 'sequential', 'none']
        :return: DataFrame with sentiment scores
        """
        try:
            mlflow.start_run()
            mlflow.log_param("device", self.device)
            mlflow.log_param("transformer_batch_size", self.transformer_batch_size)
            mlflow.log_param("sentParse", sentParse)
            mlflow.log_param("binarize", binarize)

            # self._setup_sentparser_pipeline(text_column)
            # self._setup_transformer_pipeline()
            
            # Broadcast the model and tokenizer to worker nodes
            loaded_model = self.model
            loaded_tokenizer = self.tokenizer

            def convert_to_binary_vectorized(scores):
                binary_scores = torch.zeros_like(scores)
                binary_scores[scores > 0] = 1.0
                binary_scores[scores < 0] = -1.0
                return binary_scores

            def analyze_scores_vectorized(results_tensor, binarize=False):
                # Extract the relevant scores
                label_0_scores = results_tensor[:, 0]
                label_2_scores = results_tensor[:, 2]
                
                # Compute the final score
                final_scores = label_2_scores - label_0_scores
                
                if binarize:
                    # Apply the convert_to_binary function
                    final_scores = convert_to_binary_vectorized(final_scores)
                
                return final_scores
            
            def sequential_batching(data, batch_size):
                if batch_size <= 0:
                    raise ValueError("batch_size must be a positive integer")
                
                data_array = np.array(data)
                num_batches = int(np.ceil(len(data_array) / batch_size))
                
                return [list(data_array[i*batch_size : (i+1)*batch_size]) for i in range(num_batches)]
            
            def process_in_batches(texts, model, tokenizer, accelerator, batch_size=32, max_length=16):
                # Tokenize the texts and create a dataset
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
                dataloader = DataLoader(dataset, batch_size=batch_size)

                all_scores = []

                for batch in dataloader:
                    input_ids, attention_mask = batch
                    inputs = {
                        'input_ids': input_ids.to(accelerator.device),
                        'attention_mask': attention_mask.to(accelerator.device)
                    }

                    with accelerator.autocast():  # Use mixed precision
                        with torch.no_grad():
                            outputs = model(**inputs)
                            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            all_scores.append(scores.cpu())
                            accelerator.free_memory()
                            torch.cuda.empty_cache()  # Clear CUDA cache


                return torch.cat(all_scores, dim=0)
            
            @pandas_udf(returnType=FloatType())
            def sentiment_udf(texts: pd.Series) -> pd.Series:
                model = loaded_model #model_broadcast.value
                tokenizer = loaded_tokenizer #tokenizer_broadcast.value
                accelerator = Accelerator(mixed_precision="bf16")  # Initialize the Accelerator inside the UDF
                model = accelerator.prepare(model)  # Prepare model for the accelerator
                scaler = GradScaler()  # Initialize mixed precision scaler

                if enable_batching=='dynamic':
                    # Process texts in batches
                    batch_size = 50000  # Set your desired batch size
                    scores = process_in_batches(texts.tolist(), model, tokenizer, accelerator, batch_size)
                else:
                    inputs = tokenizer(texts.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=16).to(accelerator.device)

                    with accelerator.autocast():  # Use mixed precision
                        with torch.no_grad():
                            outputs = model(**inputs)
                            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)

                final_scores_tensor = analyze_scores_vectorized(scores, binarize)
                final_scores_cpu = final_scores_tensor.cpu().numpy()
                final_scores = pd.Series(final_scores_cpu)
                accelerator.free_memory()
                return final_scores
            
            @pandas_udf(ArrayType(FloatType()))
            def sentiment_pandas_udf_array(texts: pd.Series) -> pd.Series:
                model = loaded_model #model_broadcast.value
                tokenizer = loaded_tokenizer #tokenizer_broadcast.value
                accelerator = Accelerator(mixed_precision="bf16")  # Initialize the Accelerator inside the UDF
                model = accelerator.prepare(model)  # Prepare model for the accelerator

                # Flatten the list of lists into a single list of texts
                texts = texts.reset_index(drop=True)
                all_texts = list(chain.from_iterable(texts)) 

                if enable_batching=='dynamic':
                    # Process texts in batches
                    batch_size = 20000  # Set your desired batch size
                    scores = process_in_batches(all_texts, model, tokenizer, accelerator, batch_size)
                    final_scores_tensor = analyze_scores_vectorized(scores, binarize)
                    all_sentence_scores = final_scores_tensor.cpu().numpy()
                    
                elif enable_batching=='sequential':
                    all_sentence_scores = []
                    seq_batches = sequential_batching(all_texts, 100000)
                    for batch in seq_batches:
                        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=16).to(accelerator.device)

                        with accelerator.autocast():
                            with torch.no_grad():
                                outputs = model(**inputs)
                                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        final_scores_tensor = analyze_scores_vectorized(scores, binarize)
                        final_scores_cpu = final_scores_tensor.cpu().numpy()
                        all_sentence_scores.extend(final_scores_cpu)

                else:
                    inputs = tokenizer(all_texts, return_tensors="pt", padding=True, truncation=True, max_length=16).to(accelerator.device)

                    with accelerator.autocast():  # Use mixed precision
                        with torch.no_grad():
                            outputs = model(**inputs)
                            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    final_scores_tensor = analyze_scores_vectorized(scores, binarize)
                    all_sentence_scores = final_scores_tensor.cpu().numpy()

                # Reconstruct the results to match the original structure of texts
                lengths = np.array([len(text_array) for text_array in texts])
                scores = np.split(all_sentence_scores, np.cumsum(lengths)[:-1])
                accelerator.free_memory()
                torch.cuda.empty_cache()
                return pd.Series(scores)

            if sentParse:
                if enable_batching in ('none', 'dynamic'):
                    sent_parsed_result = self.repartition_based_on_sentences(df, text_column)
                elif enable_batching == 'sequential':
                    self._setup_sentparser_pipeline(text_column)
                    sent_parser_model = self.sentParser_pipeline.fit(df)
                    sent_parsed_result = sent_parser_model.transform(df)

                torch.cuda.empty_cache()
                combined_result = sent_parsed_result.withColumn("sentence_sentiments_lst", sentiment_pandas_udf_array(col("sentences_array")))
            else:
                combined_result = df.withColumn("doc_sentiments", sentiment_udf(col(text_column)))

            # sent_df = self.reload_result(combined_result)
            combined_checkpoint = combined_result.checkpoint()
            logger.success(f"Sentence Scores loaded : {combined_result.rdd.getNumPartitions()} partitions")
            mlflow.log_metric("success", 1)
            return combined_checkpoint

        except Exception as e:
            logger.error(f"Error in sentiment inference: {e}")
            mlflow.log_metric("success", 0)
            mlflow.log_param("error", str(e))
            raise e

        finally:
            mlflow.end_run()

####################################################################################
##         Example usage of the SentimentAnalyzer class                           ##
####################################################################################

if __name__ == "__main__":

    # Setup Spark session and logging, and then execute the sentiment analysis.
    
    # from pyspark.sql import SparkSession
    # # Initialize Spark session
    # spark = SparkSession.builder \
    #     .appName("Sentiment Analysis with Transformers") \
    #     .getOrCreate()

    # Example DataFrame creation
    # data = {
    #     "reviews": [
    #         "I love the quality of this product! Don't know what to do further. But Do yo have some idea? To me, NO!!!",
    #         "This is the worst experience I ever had. But best among the current available. Whom does it matter though! ",
    #         "It was okay, not great but not terrible either. There might be better days coming. I am nobody. You can be."
    #     ]
    # }
    # df = spark.createDataFrame(pd.DataFrame(data))

    test_sentiment_df_path = '/mnt/prod/inputs/data_sources/prompt_cloud/US_haircare/ratings_and_reviews_US_haircare.parquet/'
    test_sentiment_df = spark.read.parquet(test_sentiment_df_path)
    test_sentiment_df = test_sentiment_df.withColumn("text", F.concat_ws(" . ", "ReviewTitle", "ReviewBody")).select("ReviewId", "text")

    # Create an instance of the SentimentAnalyzer
    sentiment_analyzer = SentimentAnalyzer(spark)

    # Perform sentiment analysis
    # result_df = sentiment_analyzer.trigger_SentimentInference(df, text_column="reviews", sentParse=True, binarize=False)
    result_df = sentiment_analyzer.trigger_SentimentInference(test_sentiment_df, text_column="text", sentParse=True, binarize=False)

    # Show results
    result_df.show(truncate=False)

    # Stop Spark session
    # spark.stop()

    logger.info("Execution completed. Exiting script.")
