# Spark GPU Sentiment Analyzer

Distributed sentiment analysis using **PySpark**, **Spark NLP**, and **Hugging Face Transformers**. The project targets large-scale text: GPU-friendly inference, optional quantization and mixed precision, sentence- and document-level scores, and **MLflow** for experiment tracking. It fits batch pipelines on Spark clusters (including Databricks-style layouts) where you need throughput without giving up modern transformer quality.

## Tech stack

- **PySpark** — distributed DataFrames and orchestration
- **Spark NLP** — document assembly, sentence segmentation
- **Hugging Face Transformers** — model and tokenizer loading
- **RoBERTa** (and other compatible classifiers) — sentiment heads
- **MLflow** — parameters, metrics, and run metadata
- **CUDA** — optional GPU acceleration when a compatible PyTorch build is installed

## Features

- Distributed text processing with **PySpark** for large datasets
- **Sentence-level** analysis via Spark NLP’s sentence detector
- **Transformer models** (e.g. RoBERTa) from Hugging Face for sentiment
- **GPU acceleration**, **quantization**, and **mixed precision (bf16 / fp16)** where supported
- Dynamic batching and memory-conscious inference patterns
- **MLflow** integration for logging and comparison across runs

## Architecture overview

1. **Data loading** — Parquet, CSV, or other Spark sources into DataFrames
2. **Sentence parsing** — Spark NLP **SentenceDetector** (optional) for sentence-level scores
3. **Model setup** — Hugging Face model + tokenizer, with optional quantization and mixed precision
4. **Batching** — Dynamic or sequential batches tuned to data size and hardware
5. **Inference** — Forward passes to produce sentiment labels and/or scores
6. **Tracking** — MLflow records parameters, metrics, and artifacts per run

## Quick start

### Prerequisites

- Java and a Spark-compatible environment for PySpark
- **Spark NLP** JARs / packages as required by your cluster (see [Spark NLP docs](https://nlp.johnsnowlabs.com/docs/en/install))
- **MLflow** tracking URI and experiment name configured for your deployment
- For GPU: a **CUDA**-capable setup and a PyTorch build that matches your driver

### Installation

Clone the repository and install Python dependencies:

```bash
git clone https://github.com/anmolg1997/Spark-GPU-Sentiment-Analyzer.git
cd Spark-GPU-Sentiment-Analyzer
pip install -r requirements.txt
```

The bundled scripts (`sentimentAnalyzer_spark_gpu_databricks.py`, `sentimentAnalyzer (3).py`) also use packages such as **loguru** and **VADER**; install them if you run those entry points as-is:

```bash
pip install loguru vaderSentiment
```

### Configuration

- Set **MODEL_DIRECTORY** (and related paths) in the script to your Hugging Face or local model location
- Point **MLflow** at your experiment name and tracking server
- Tune **batch size**, **max sequence length**, and device settings for your cluster

### Running the scripts

From the repository root, you can import the `SentimentAnalyzer` class defined in `sentimentAnalyzer_spark_gpu_databricks.py` (primary GPU-oriented reference) or use `sentimentAnalyzer (3).py` depending on your environment. Example pattern:

```python
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from sentimentAnalyzer_spark_gpu_databricks import SentimentAnalyzer

spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()

df = spark.read.parquet("/path/to/your/data")

sentiment_analyzer = SentimentAnalyzer(spark)
result_df = sentiment_analyzer.trigger_SentimentInference(
    df, text_column="text", sentParse=True
)
result_df.show(truncate=False)
```

Adjust `text_column`, paths, and Spark session builder options for your platform (local, EMR, Databricks, etc.).

### Example usage

```python
if __name__ == "__main__":
    spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()
    test_sentiment_df = spark.read.parquet("/mnt/prod/inputs/data_sources/reviews.parquet")
    test_sentiment_df = test_sentiment_df.withColumn(
        "text", F.concat_ws(" . ", "ReviewTitle", "ReviewBody")
    )

    sentiment_analyzer = SentimentAnalyzer(spark)
    result_df = sentiment_analyzer.trigger_SentimentInference(
        test_sentiment_df, text_column="text", sentParse=True
    )
    result_df.show(truncate=False)
```

Supporting assets in the repo include `Download & Save _ Huggingface Models.py` for caching models locally and `RoBERTa - Text Classifier Framework.ipynb` for exploratory RoBERTa work.

## Performance optimizations

- **GPU** — Uses available accelerators when PyTorch is built with CUDA
- **Quantization** — e.g. 4-bit paths where configured, to shrink memory and speed up inference
- **Mixed precision** — bf16 / fp16 for faster matmuls where numerically stable
- **Dynamic batching** — Adapts batch size to workload and memory headroom

## Logging and experiment tracking

Runs can be captured in **MLflow** (parameters, metrics, logs, and errors) so you can compare configurations and model versions over time.

## Customization

- Point the pipeline at a different Hugging Face model by changing the model path in `SentimentAnalyzer`
- Switch batching behavior via parameters such as `enable_batching` where exposed in the script

## Future work

- Multiple transformer models for ensemble sentiment
- Broader multi-class sentiment taxonomies
- Tighter integration with streaming or near-real-time ingestion

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
