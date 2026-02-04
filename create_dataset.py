from dotenv import load_dotenv
from langfuse import get_client
from groq import Groq
import json
from langfuse import get_client

load_dotenv()

groq_client = Groq()


# =============================================================================
# CREATING DATASETS
# =============================================================================

def create_sentiment_dataset():
    """
    Create a dataset for testing sentiment analysis.
    In practice, you'd often populate this from production data or CSV.
    """

    # Create or get existing dataset
    try:
        # Try to get existing dataset
        dataset = get_client().get_dataset(name="sentiment-benchmark-v1")
        print("✓ Using existing dataset")
    except:
        # Create new dataset if it doesn't exist
        dataset = get_client().create_dataset(
            name="sentiment-benchmark-v1",
            description="Benchmark dataset for sentiment analysis evaluation",
            metadata={
                "created_by": "lecture_demo",
                "domain": "product_reviews",
                "version": "1.0"
            }
        )

        # Test cases with ground truth
        test_cases = [
            {
                "input": {"text": "This product is absolutely amazing! Best purchase ever!"},
                "expected_output": {"sentiment": "positive", "confidence_min": 0.8},
                "metadata": {"category": "enthusiastic_positive"}
            },
            {
                "input": {"text": "Terrible experience. Product broke after one day."},
                "expected_output": {"sentiment": "negative", "confidence_min": 0.8},
                "metadata": {"category": "strong_negative"}
            },
            {
                "input": {"text": "It's okay, nothing special but does the job."},
                "expected_output": {"sentiment": "neutral", "confidence_min": 0.5},
                "metadata": {"category": "neutral"}
            },
            {
                "input": {"text": "The quality is good but the price is too high."},
                "expected_output": {"sentiment": "mixed", "confidence_min": 0.5},
                "metadata": {"category": "mixed_sentiment"}
            },
            {
                "input": {"text": "Exceeded expectations! Will definitely buy again."},
                "expected_output": {"sentiment": "positive", "confidence_min": 0.85},
                "metadata": {"category": "positive_with_intent"}
            },
            {
                "input": {"text": "Not what I expected. Returning it tomorrow."},
                "expected_output": {"sentiment": "negative", "confidence_min": 0.7},
                "metadata": {"category": "negative_with_action"}
            },
            {
                "input": {"text": "Average product. Works as described."},
                "expected_output": {"sentiment": "neutral", "confidence_min": 0.6},
                "metadata": {"category": "factual_neutral"}
            },
            {
                "input": {"text": "I HATE THIS!!! Worst purchase of my life!!!"},
                "expected_output": {"sentiment": "negative", "confidence_min": 0.95},
                "metadata": {"category": "extreme_negative"}
            },
        ]

        # Add items to dataset
        for i, case in enumerate(test_cases):
            get_client().create_dataset_item(
                dataset_name="sentiment-benchmark-v1",
                input=case["input"],
                expected_output=case["expected_output"],
                metadata=case["metadata"],
            )

        print(f"✓ Created dataset with {len(test_cases)} test cases")
    
    return dataset


def sentiment_task(text:str)->dict:
    reponse=groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{
            "role":"system",
            "content":""" Analyze sentiment and respond in JSON:
            {
            "sentiment":"positive|negative|neutral|mixed",
            "confidence":0.0-1.0,
            "reasoning":"brief explanation"
            }"""},
            {"role":"user","content":text}
            ],
        temperature=0.2
    )
    return json.loads(reponse.choices[0].message.content)


def simple_evaluator(output:dict,expected:dict)->dict:
    """
    Evaluate the sentiment task output against expected values.

    Args:
        output (dict): Output from sentiment_task with keys: sentiment, confidence, reasoning
        expected (dict): Expected output with keys: sentiment, confidence_min

    Returns:
        dict: Evaluation result with pass/fail and details

    Example:
        {"sentiment": "negative", "confidence_min": 0.95}
    """
    sentiment_match = output.get("sentiment") == expected.get("sentiment")
    confidence_ok = output.get("confidence", 0) >= expected.get("confidence_min", 0)
    passed = sentiment_match and confidence_ok
    
    return {
        "passed": passed,
        "sentiment_match": sentiment_match,
        "confidence_ok": confidence_ok,
        "predicted_sentiment": output.get("sentiment"),
        "expected_sentiment": expected.get("sentiment"),
        "predicted_confidence": output.get("confidence"),
        "required_confidence": expected.get("confidence_min")
    }


def evaluate_on_dataset():
    """
    Test sentiment_task on all items in the dataset and evaluate with simple_evaluator.
    """
    client = get_client()
    
    # Get the dataset
    dataset = client.get_dataset(name="sentiment-benchmark-v1")
    
    # Get all items from the dataset
    dataset_items = dataset.items
    
    print("=" * 80)
    print("EVALUATING SENTIMENT TASK ON DATASET")
    print("=" * 80)
    
    results = []
    passed_count = 0
    
    for i, item in enumerate(dataset_items, 1):
        # Run sentiment task on the input text
        text = item.input.get("text")
        print(f"\n[Test {i}] Input: {text[:60]}...")
        
        try:
            # Get the sentiment analysis
            output = sentiment_task(text)
            print(f"  Output: {output}")
            
            # Evaluate the output
            evaluation = simple_evaluator(output, item.expected_output)
            results.append(evaluation)
            
            # Display results
            status = "✓ PASS" if evaluation["passed"] else "✗ FAIL"
            print(f"  {status}")
            print(f"    Sentiment: {evaluation['predicted_sentiment']} (expected: {evaluation['expected_sentiment']}) - {'✓' if evaluation['sentiment_match'] else '✗'}")
            print(f"    Confidence: {evaluation['predicted_confidence']:.2f} (required: ≥{evaluation['required_confidence']}) - {'✓' if evaluation['confidence_ok'] else '✗'}")
            
            if evaluation["passed"]:
                passed_count += 1
                
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            results.append({"passed": False, "error": str(e)})
    
    # Summary
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed_count}/{len(results)} tests passed ({100*passed_count//len(results)}%)")
    print("=" * 80)
    
    return results


create_sentiment_dataset()
evaluate_on_dataset()


