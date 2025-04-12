"""
Example demonstrating result validation.

This example shows how to validate LLM-generated results against
custom criteria and handle cases where validation fails.
"""

import asyncio
import os
from typing import Optional

from aikernel import LLMMessagePart, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import Field, field_validator

from goose import Agent, FlowArguments, Result, flow, task


class ProductReview(Result):
    """A product review with ratings and analysis."""
    product_name: str = Field(description="Name of the product being reviewed")
    overall_rating: int = Field(description="Overall rating from 1-5 stars")
    pros: list[str] = Field(description="Positive aspects of the product")
    cons: list[str] = Field(description="Negative aspects of the product")
    summary: str = Field(description="Brief summary of the review")
    
    @field_validator("overall_rating")
    @classmethod
    def validate_rating(cls, value: int) -> int:
        """Validate that the rating is between 1 and 5."""
        if value < 1 or value > 5:
            raise ValueError("Rating must be between 1 and 5")
        return value


class ProductReviewResult(Result):
    """Result of a product review analysis."""
    review: ProductReview = Field(description="The product review")
    sentiment_score: float = Field(description="Sentiment score from -1.0 (negative) to 1.0 (positive)")
    key_insights: list[str] = Field(description="Key insights from the review")
    
    @field_validator("sentiment_score")
    @classmethod
    def validate_sentiment_score(cls, value: float) -> float:
        """Validate that the sentiment score is between -1.0 and 1.0."""
        if value < -1.0 or value > 1.0:
            raise ValueError("Sentiment score must be between -1.0 and 1.0")
        return value


class ValidationFlowArguments(FlowArguments):
    """Arguments for the review analysis flow."""
    product_info: str


@task
async def analyze_reviews(*, agent: Agent, product_info: str) -> ProductReviewResult:
    """Analyze product reviews and extract insights."""
    print(f"Analyzing reviews for: {product_info}")
    
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content="You are a product review analyst. Create a detailed review analysis based on the product information.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"Please analyze reviews for this product: {product_info}\n\nCreate a detailed review with ratings, pros, cons, and a summary. Then provide a sentiment score between -1.0 and 1.0, and key insights.")]
    )
    
    # Make the actual LLM call with validation
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await agent(
                messages=[system_message, user_message],
                model="gemini-2.0-flash",
                task_name="analyze_reviews",
                response_model=ProductReviewResult,
                router=router
            )
        except ValueError as e:
            if attempt < max_retries - 1:
                print(f"Validation failed: {e}. Retrying ({attempt + 1}/{max_retries})...")
                
                # Add feedback about the validation error
                feedback_message = LLMUserMessage(
                    parts=[LLMMessagePart(content=f"The previous response failed validation: {e}. Please ensure the sentiment score is between -1.0 and 1.0, and the overall rating is between 1 and 5.")]
                )
                
                # Add the feedback to the conversation
                user_message = feedback_message
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                raise


@flow
async def review_analysis_flow(*, flow_arguments: ValidationFlowArguments, agent: Agent) -> None:
    """Flow that analyzes product reviews with validation."""
    await analyze_reviews(agent=agent, product_info=flow_arguments.product_info)


async def main():
    """Run the review analysis flow and demonstrate validation."""
    # Create a unique run ID
    run_id = f"review-analysis-{os.getpid()}"
    
    print("=== Result Validation Example ===")
    print("This example demonstrates how to validate LLM-generated results")
    print("against custom criteria and handle cases where validation fails.\n")
    
    # Sample product to analyze
    product_info = """
    Wireless Noise-Cancelling Headphones XZ-500
    
    Features:
    - Active noise cancellation
    - 30-hour battery life
    - Bluetooth 5.0 connectivity
    - Built-in microphone for calls
    - Foldable design
    - Available in black, white, and blue
    
    Price: $199.99
    
    Recent customer feedback mentions good sound quality but some issues with comfort during extended use.
    """
    
    # Run the review analysis flow
    async with review_analysis_flow.start_run(run_id=run_id) as run:
        await review_analysis_flow.generate(ValidationFlowArguments(product_info=product_info))
        
        # Get the analysis result
        result = run.get_result(task=analyze_reviews)
        
        print("\n--- Review Analysis Results ---")
        print(f"Product: {result.review.product_name}")
        print(f"Overall Rating: {result.review.overall_rating}/5 stars")
        
        print("\nPros:")
        for pro in result.review.pros:
            print(f"- {pro}")
        
        print("\nCons:")
        for con in result.review.cons:
            print(f"- {con}")
        
        print(f"\nSummary: {result.review.summary}")
        
        print(f"\nSentiment Score: {result.sentiment_score:.2f}")
        
        print("\nKey Insights:")
        for insight in result.key_insights:
            print(f"- {insight}")
        
        print("\nNote: This example demonstrates how Goose can validate LLM-generated")
        print("results against custom criteria using Pydantic validators, and")
        print("automatically retry when validation fails.")


if __name__ == "__main__":
    asyncio.run(main())
