
# Sentiment Analysis with PySpark and Transformers

This project implements a scalable and efficient sentiment analysis framework using **PySpark**, **Spark NLP**, and **Hugging Face Transformers**. It is designed to process large-scale textual data in a distributed environment, leveraging **GPU acceleration**, **quantized models**, and **mixed precision** techniques for faster and more accurate sentiment analysis. This framework is ideal for handling extensive datasets in real-time and producing detailed sentiment scores at both the sentence and document levels.

## Features

- Distributed text processing using **PySpark** for handling large datasets.
- **Sentence-level sentiment analysis** using **Spark NLP** for granular results.
- Supports **state-of-the-art transformer models** (e.g., RoBERTa) from Hugging Face for sentiment analysis.
- Optimized for **GPU acceleration**, including **quantization** and **mixed precision (bf16/float16)** for faster inference.
- Dynamic batching and efficient memory management to improve inference speed.
- Comprehensive logging and experiment tracking using **MLflow**.

## Architecture Overview

1. **Data Loading**: Load data (e.g., product reviews) from Parquet or CSV files into PySpark DataFrames.
2. **Sentence Parsing**: Use Spark NLP’s **SentenceDetector** to split text into sentences for detailed sentiment analysis.
3. **Transformer Model Setup**: Load pre-trained or fine-tuned transformer models and tokenizers from Hugging Face, optimized with **quantization** and **mixed precision**.
4. **Batching Strategy**: Apply dynamic or sequential batching based on the dataset and resource availability.
5. **Sentiment Inference**: Process text through the transformer model to generate sentiment scores.
6. **Logging and Monitoring**: Track all processes, metrics, and parameters using **MLflow** for experiment tracking.

## Quick Start

### Prerequisites

1. **PySpark**: Ensure PySpark is installed and set up correctly for distributed processing.
2. **Hugging Face Transformers**: Install the Hugging Face transformers library for NLP models.
3. **Spark NLP**: Required for sentence parsing and NLP tasks.
4. **MLflow**: For tracking experiments and logging metrics.
5. **CUDA**: If running on GPU, ensure CUDA is properly configured.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/sentiment-analysis-pyspark-transformers.git
   cd sentiment-analysis-pyspark-transformers
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

- Set up the **MODEL_DIRECTORY** in the script to the location of your pre-trained transformer model.
- Update the **MLflow experiment name** to your desired experiment folder.
- Customize **batch size**, **max sequence length**, and other hyperparameters based on your dataset.

### Running the Script

1. Initialize a Spark session and load your data:

   ```python
   from pyspark.sql import SparkSession
   spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()
   
   # Load data into a DataFrame (from Parquet/CSV)
   df = spark.read.parquet("/path/to/your/data")
   ```

2. Create an instance of the `SentimentAnalyzer` class and run the sentiment analysis:

   ```python
   from sentiment_analyzer import SentimentAnalyzer
   
   sentiment_analyzer = SentimentAnalyzer(spark)
   result_df = sentiment_analyzer.trigger_SentimentInference(df, text_column="text", sentParse=True)
   
   # Show results
   result_df.show(truncate=False)
   ```

3. Once the analysis is complete, the results are saved and logged in **MLflow**.

### Example Usage

```python
if __name__ == "__main__":
    spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()
    test_sentiment_df = spark.read.parquet("/mnt/prod/inputs/data_sources/reviews.parquet")
    test_sentiment_df = test_sentiment_df.withColumn("text", F.concat_ws(" . ", "ReviewTitle", "ReviewBody"))
    
    sentiment_analyzer = SentimentAnalyzer(spark)
    result_df = sentiment_analyzer.trigger_SentimentInference(test_sentiment_df, text_column="text", sentParse=True)
    result_df.show(truncate=False)
```

## Performance Optimizations

- **GPU Acceleration**: Automatically detects and leverages GPU for faster inference.
- **Quantization**: Uses **4-bit quantization** to reduce the memory footprint and speed up transformer models.
- **Mixed Precision**: Applies **mixed precision (bf16/float16)** for faster computations without sacrificing accuracy.
- **Dynamic Batching**: Adjusts batch size based on dataset size and system resources for optimized processing.

## Logging and Experiment Tracking

This framework integrates **MLflow** for experiment tracking. Parameters, metrics, and logs (including errors) are automatically recorded for each run.

## Customization

- You can change the transformer model by updating the model path in the `SentimentAnalyzer` class.
- Modify the batching strategy (dynamic, sequential, or no batching) by adjusting the `enable_batching` parameter.

## Future Work

- Support for **multiple transformer models** for ensemble sentiment analysis.
- Extending the framework for multi-class sentiment classification.
- Integration with real-time data pipelines for live sentiment analysis.

## Contact

For further information or inquiries, feel free to contact **Anmol Jaiswal** at anmol@i-genie.ai.
