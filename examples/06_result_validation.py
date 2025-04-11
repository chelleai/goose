"""
Example demonstrating result validation.

This example shows how Goose validates model outputs against expected schemas
and handles validation errors appropriately.
"""

import asyncio
import json
from typing import List, Optional

from aikernel import LLMModelAlias, LLMRouter, LLMSystemMessage, LLMUserMessage
from pydantic import BaseModel, Field, ValidationError

from goose import Agent, FlowArguments, Result, flow, task


class ProductReview(BaseModel):
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")


class ProductReviewResult(Result):
    product_name: str = Field(description="Name of the reviewed product")
    reviews: List[ProductReview] = Field(description="List of product reviews")
    average_rating: float = Field(description="Average rating of all reviews")
    recommendation: str = Field(description="Recommendation based on reviews")


class ValidationFlowArguments(FlowArguments):
    product_description: str


@task
async def analyze_reviews(*, agent: Agent, product_info: str) -> ProductReviewResult:
    """Analyze product reviews and return a structured result."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    return await agent(
        messages=[
            LLMSystemMessage(content="""
                You are a product review analyst.
                Analyze the product information and create a structured analysis with reviews.
                The rating must be between 1 and 5 (inclusive).
            """),
            LLMUserMessage(content=f"Analyze this product information and generate reviews:\n\n{product_info}")
        ],
        model="gemini-2.0-flash",
        task_name="analyze_reviews",
        response_model=ProductReviewResult,
        router=router
    )


# Deliberate error case to demonstrate validation
async def trigger_validation_error(agent: Agent) -> None:
    """Deliberately trigger a validation error to demonstrate how it's handled."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    # Create an invalid response that will fail validation
    invalid_json = {
        "product_name": "Smartphone X",
        "reviews": [
            {
                "rating": 10,  # Invalid: rating should be 1-5
                "comment": "Great product!",
                "pros": ["Fast", "Good camera"],
                "cons": ["Expensive"]
            }
        ],
        "average_rating": 4.5,
        "recommendation": "Recommended"
    }
    
    try:
        # Try to parse invalid JSON directly (without the LLM)
        ProductReviewResult.model_validate(invalid_json)
    except ValidationError as e:
        print("VALIDATION ERROR CAUGHT:")
        for error in e.errors():
            print(f"- {error['loc']}: {error['msg']}")
        
        # The framework would handle this automatically with real LLM responses


@flow
async def review_analysis_flow(*, flow_arguments: ValidationFlowArguments, agent: Agent) -> None:
    """Flow for analyzing product reviews with validation."""
    await analyze_reviews(agent=agent, product_info=flow_arguments.product_description)


async def main():
    product_info = """
    Smartphone X1 Pro:
    - 6.7-inch OLED display
    - 128GB storage
    - 12GB RAM
    - Triple camera system (50MP main, 12MP ultrawide, 5MP macro)
    - 5000mAh battery
    - Android 14
    - $999 price point
    - Available in black, silver, and blue
    """
    
    # Run the validation flow
    async with review_analysis_flow.start_run(run_id="review-analysis-1") as run:
        try:
            await review_analysis_flow.generate(ValidationFlowArguments(product_description=product_info))
            
            # Display the validated result
            review_result = run.get_result(task=analyze_reviews)
            print("\nVALIDATED RESULT:")
            print("=" * 50)
            print(f"Product: {review_result.product_name}")
            print(f"Average Rating: {review_result.average_rating}/5")
            print(f"Recommendation: {review_result.recommendation}")
            print("\nReviews:")
            
            for i, review in enumerate(review_result.reviews, 1):
                print(f"\nReview #{i}:")
                print(f"Rating: {'⭐' * review.rating} ({review.rating}/5)")
                print(f"Comment: {review.comment}")
                print("Pros:")
                for pro in review.pros:
                    print(f"- {pro}")
                print("Cons:")
                for con in review.cons:
                    print(f"- {con}")
            
            # Demonstrate validation error handling
            print("\n" + "=" * 50)
            print("DEMONSTRATING VALIDATION ERROR HANDLING:")
            await trigger_validation_error(agent=run.agent)
            
        except Exception as e:
            print(f"Error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())