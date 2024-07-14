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
Last updated: 22 Jun' 2024
Organization: i-Genie
Version: 1.0
"""

####################################################################################
##                            Importing Modules & Libraries                       ##
####################################################################################
import time
from loguru import logger

import numpy as np
import pandas as pd

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col, explode, pandas_udf, udf, posexplode, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType, ArrayType

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import SentenceDetector

####################################################################################
##                               Configuration Setup                              ##
####################################################################################
# Constants for model configurations and file paths.
MODEL_DIRECTORY = "/dbfs/mnt/igenie-blob01/Anmol_AI_dir/model_dir"

# Define label mappings to standardize output labels across different sentiment models
label_mappings = {
    "distilbert": {},
    "bert": {
        "1 star": "NEGATIVE",
        "2 stars": "NEGATIVE",
        "3 stars": "NEUTRAL",
        "4 stars": "POSITIVE",
        "5 stars": "POSITIVE"
    },
    "roberta": {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE"
    },
    "albert": {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "POSITIVE"
    }
}

####################################################################################
##                           Sentiment Analyzer Class                             ##
####################################################################################
class SentimentAnalyzer:
    def __init__(self, spark_session, transformer_batch_size=1028, device=None):
        """
        Initialize the Sentiment Analyzer class configured for transformer-based models.
        :param spark_session: Active Spark session.
        :param transformer_batch_size: Batch size for processing text in transformers.
        :param device: Computation device ('cpu' or 'gpu') or None for auto-detection.
        """
        assert transformer_batch_size > 0, "Batch size must be positive"
        self.spark = spark_session
        self.transformer_batch_size = transformer_batch_size
        self.device = self._get_device(device)
        self.models_lst = ["distilbert", "bert", "roberta", "albert"]
        self.broadcast_label_mappings = self.spark.sparkContext.broadcast(label_mappings)
        logger.info("SentimentAnalyzer initialized with batch size {} on device cuda:{}".format(transformer_batch_size, self.device))

    def _get_device(self, device):
        """
        Determine the computation device based on availability or preference.
        :return: Device identifier suitable for transformer operations.
        """
        device_selection = 'GPU' if torch.cuda.is_available() else 'CPU'
        logger.info("Computation will be performed on: {}".format(device_selection))
        return 0 if device == 'gpu' else -1

    def _setup_sentparser_pipeline(self, text_column: str):
        """
        Configure the Spark NLP pipeline to extract sentences from a text column.
        :param text_column: Column name containing text data.
        """
        logger.info("Setting up the Spark NLP sentence parsing pipeline for column: {}".format(text_column))
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
        logger.info("Spark NLP pipeline configured successfully.")

    def _setup_transformer_pipeline(self, model_name: str):
        """
        Load and configure the transformer pipeline for sentiment analysis.
        :param model_name: Name of the transformer model to use.
        """
        logger.info("Loading and setting up transformer pipeline for model: {}".format(model_name))
        model_save_path = f"{MODEL_DIRECTORY}/{model_name}-model"
        tokenizer_save_path = f"{MODEL_DIRECTORY}/{model_name}-tokenizer"

        if model_name not in self.models_lst:
            logger.error(f"Invalid model name: {model_name}")
            raise ValueError(f"Invalid model name: {model_name}")

        loaded_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
        loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            use_fast=True,
            padding=True,
            truncation=True,
            max_length=512,
            device=self.device,
            batch_size=self.transformer_batch_size
        )
        logger.info("Transformer sentiment pipeline for {} is configured and ready.".format(model_name))

    def _setup_vader_pipeline(self):
        """
        Setup the VADER sentiment analysis tool.
        """
        self.vader_analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER sentiment analysis pipeline setup completed")

    def trigger_SentimentInference(self, df: DataFrame, model_name='roberta', text_column="text", sentParse=True, binarize=False):
        """
        Trigger sentiment analysis on a DataFrame using specified model and settings.

        :param df: DataFrame containing the text data.
        :param model_name: Model name to use for sentiment analysis.
        :param text_column: Text column for sentiment analysis.
        :param sentParse: Boolean indicating whether to parse sentences.
        :param binarize: Boolean indicating whether to binarize scores.
        :return: DataFrame with sentiment scores added.
        """
        self._setup_sentparser_pipeline(text_column)

        if model_name.lower() == 'vader':
            self._setup_vader_pipeline()
            loaded_sentiment_analyzer = self.vader_analyzer

            @pandas_udf(ArrayType(FloatType()))
            def vader_sentiment_pandas_udf_array(sentences: pd.Series) -> pd.Series:
                """
                A pandas UDF to apply VADER sentiment analysis to a series of sentences, optionally binarizing the result.

                :param sentences: A pandas Series of sentences.
                :return: A Series of sentiment scores.
                """
                convert_to_binary = lambda score: 1.0 if score > 0 else -1.0 if score < 0 else 0.0
                def analyze_sentence(sentence):
                    vs = loaded_sentiment_analyzer.polarity_scores(sentence)
                    score = vs['compound']
                    return convert_to_binary(score) if binarize else score
                return sentences.apply(lambda x: [analyze_sentence(sentence) for sentence in x])
            logger.debug("VADER sentiment UDF setup completed for array processing")

            @pandas_udf(returnType=DoubleType())
            def vader_sentiment_udf(texts):
                """
                A pandas UDF to apply VADER sentiment analysis to a series of texts, returning a compound sentiment score.

                :param texts: A pandas Series of text documents.
                :return: A Series of sentiment scores.
                """
                results = texts.apply(lambda text: loaded_sentiment_analyzer.polarity_scores(text)['compound'])
                convert_to_binary = lambda score: 1.0 if score > 0 else -1.0 if score < 0 else 0.0
                if binarize:
                    results = results.apply(convert_to_binary)
                return results.astype(float)
            logger.debug("VADER sentiment UDF setup completed for document-level analysis")

            if sentParse:
                sent_parser_model = self.sentParser_pipeline.fit(df)
                sent_parsed_result = sent_parser_model.transform(df)
                final_result = sent_parsed_result.withColumn("sentence_sentiments_lst", vader_sentiment_pandas_udf_array(col("sentences_array")))
                logger.info("Sentiment analysis triggered with sentence parsing using VADER")
            else:
                final_result = df.withColumn("doc_sentiments", vader_sentiment_udf(col(text_column)))
                logger.info("Sentiment analysis triggered without sentence parsing using VADER")
        else:
            self._setup_transformer_pipeline(model_name)
            loaded_sentiment_analyzer = self.sentiment_pipeline

            label_mappings_value = self.broadcast_label_mappings.value

            @pandas_udf(ArrayType(FloatType()))
            def sentiment_pandas_udf_array(sentences: pd.Series) -> pd.Series:
                """
                A pandas UDF to apply transformer-based sentiment analysis to a series of sentences, with an option to binarize the results.

                :param sentences: A pandas Series of sentences.
                :return: A Series of transformed sentiment scores.
                """
                convert_to_binary = lambda score: 1.0 if score > 0 else -1.0 if score < 0 else 0.0
                def analyze_sentence(sentence):
                    result_sent = loaded_sentiment_analyzer(sentence)[0]
                    label = result_sent['label']
                    label = label_mappings_value[model_name].get(label, label).lower()
                    score = result_sent['score']
                    score = -score if 'negative' in label else score - 0.5 if 'neutral' in label else score
                    return convert_to_binary(score) if binarize else score
                result_lst = sentences.apply(lambda x: [analyze_sentence(sentence) for sentence in x])
                return result_lst

            @pandas_udf(returnType=DoubleType())
            def sentiment_udf(texts):
                """
                A pandas UDF to apply transformer-based sentiment analysis to a list of texts, returning detailed sentiment scores.

                :param texts: A list of text documents.
                :return: A Series of sentiment scores, potentially binarized.
                """
                texts = texts.tolist()
                results = loaded_sentiment_analyzer(texts)
                df_results = pd.DataFrame(results)
                convert_to_binary = lambda score: 1.0 if score > 0 else -1.0 if score < 0 else 0.0

                if label_mappings_value[model_name] is not None:
                    df_results['label'] = df_results['label'].map(label_mappings_value[model_name]).fillna(df_results['label'])
                
                labels = df_results['label'].str.lower()
                scores = df_results['score']
                
                negative_mask = labels.str.contains('negative')
                positive_mask = labels.str.contains('positive')
                neutral_mask = labels.str.contains('neutral')
                
                df_results['score'] = np.where(negative_mask, -scores, np.where(positive_mask, scores, np.where(neutral_mask, scores-0.5, scores)))
                if binarize:
                    df_results['score'] = df_results['score'].apply(convert_to_binary)
                return df_results['score'].astype(float)

            if sentParse:
                sent_parser_model = self.sentParser_pipeline.fit(df)
                sent_parsed_result = sent_parser_model.transform(df)
                logger.debug("Transformer-based sentiment UDF setup completed for array processing")
                final_result = sent_parsed_result.withColumn("sentence_sentiments_lst", sentiment_pandas_udf_array(col("sentences_array")))
                logger.info("Sentiment analysis triggered with sentence parsing using transformer model")
            elif not sentParse:
                logger.debug("Transformer-based sentiment UDF setup completed for document-level analysis")
                final_result = df.withColumn("doc_sentiments", sentiment_udf(col(text_column)))
                logger.info("Sentiment analysis triggered without sentence parsing using transformer model")

        logger.success("Sentence Score for Array Added")
        return final_result

    def trigger_SentimentInference_old(self, df: DataFrame, model_name='roberta', text_column="text", sentParse=True):
        """
        Perform sentiment analysis on DataFrame text data.
        :param df: DataFrame containing text.
        :param model_name: Transformer model to use.
        :param text_column: Text data column.
        :param sentParse: Boolean to determine if text is parsed into sentences before analysis.
        :return: DataFrame with appended sentiment analysis results.
        """
        logger.info("Starting sentiment analysis for model: {}".format(model_name))
        self._setup_sentparser_pipeline(text_column)
        self._setup_transformer_pipeline(model_name)
        loaded_sentiment_analyzer = self.sentiment_pipeline
        label_mappings_value = self.broadcast_label_mappings.value

        # Define Pandas UDFs for sentiment analysis
        @pandas_udf(ArrayType(FloatType()))
        def sentiment_pandas_udf_array(sentences: pd.Series) -> pd.Series:
            """
            Compute sentiment scores for an array of sentences using a pandas UDF.
            :param sentences: Series of text sentences.
            :return: Series of sentiment scores.
            """
            def analyze_sentence(sentence):
                result_sent = loaded_sentiment_analyzer(sentence)[0]
                label = result_sent['label']
                label = label_mappings_value[model_name].get(label, label).lower()
                score = result_sent['score']
                if 'negative' in label:
                    score = -score
                elif 'positive' in label:
                    score = score
                elif 'neutral' in label:
                    score = score - 0.5
                return score

            result_lst = sentences.apply(lambda x: [analyze_sentence(sentence) for sentence in x])
            return result_lst

        @pandas_udf(DoubleType())
        def sentiment_udf(texts: pd.Series) -> pd.Series:
            """
            Compute sentiment scores for a batch of text using a pandas UDF.
            :param texts: Series of text data.
            :return: Series of adjusted sentiment scores based on polarity.
            """
            texts = texts.tolist()
            results = loaded_sentiment_analyzer(texts)
            df_results = pd.DataFrame(results)
            if label_mappings_value[model_name] is not None:
                df_results['label'] = df_results['label'].map(label_mappings_value[model_name]).fillna(df_results['label'])
            labels = df_results['label'].str.lower()
            scores = df_results['score']
            negative_mask = labels.str.contains('negative')
            positive_mask = labels.str.contains('positive')
            neutral_mask = labels.str.contains('neutral')
            df_results['score'] = np.where(negative_mask, -scores, np.where(positive_mask, scores, np.where(neutral_mask, scores-0.5, scores)))
            result_lst = df_results['score'].astype(float)
            return result_lst

        # Execute sentiment analysis and return results
        if sentParse:
            logger.info("Parsing sentences and performing sentiment analysis.")
            sent_parser_model = self.sentParser_pipeline.fit(df)
            sent_parsed_result = sent_parser_model.transform(df)
            final_result = sent_parsed_result.withColumn("sentence_sentiments_lst", sentiment_pandas_udf_array(col("sentences_array")))
        else:
            logger.info("Performing document-level sentiment analysis.")
            final_result = df.withColumn("doc_sentiments", sentiment_udf(col(text_column)))

        logger.success("Sentiment analysis process completed successfully.")
        return final_result


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
    data = {
        "reviews": [
            "I love the quality of this product! Don't know what to do further.",
            "This is the worst experience I ever had. But best among the current available.",
            "It was okay, not great but not terrible either. There might be better days coming."
        ]
    }
    df = spark.createDataFrame(pd.DataFrame(data))

    # Create an instance of the SentimentAnalyzer
    sentiment_analyzer = SentimentAnalyzer(spark, transformer_batch_size=1028, device='gpu')

    # Perform sentiment analysis
    result_df = sentiment_analyzer.trigger_SentimentInference(df, model_name='roberta', text_column="reviews", sentParse=True, binarize=False)

    # Show results
    result_df.show(truncate=False)

    # Stop Spark session
    # spark.stop()

    logger.info("Execution completed. Exiting script.")
