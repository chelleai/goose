"""
Example demonstrating result validation.

This example shows how Goose validates model outputs against expected schemas
and handles validation errors appropriately.

NOTE: This is a demo file that illustrates the pattern without making actual LLM calls.
"""

import asyncio
import os
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

from goose import Agent, FlowArguments, Result, flow, task


class ProductReview(BaseModel):
    """Product review with rating, comment, pros and cons."""
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")


class ProductReviewResult(Result):
    """Aggregated review analysis for a product."""
    product_name: str = Field(description="Name of the reviewed product")
    reviews: List[ProductReview] = Field(description="List of product reviews")
    average_rating: float = Field(description="Average rating of all reviews")
    recommendation: str = Field(description="Recommendation based on reviews")


class ValidationFlowArguments(FlowArguments):
    """Arguments for the review analysis flow."""
    product_description: str


# Mock implementation for review analysis
def mock_review_analysis(product_info: str) -> ProductReviewResult:
    """Generate a mock review analysis based on the product info."""
    if "Smartphone" in product_info:
        return ProductReviewResult(
            product_name="Smartphone X1 Pro",
            reviews=[
                ProductReview(
                    rating=5,
                    comment="Excellent phone with amazing camera quality and battery life!",
                    pros=["Stunning display", "Excellent camera system", "All-day battery life", "Premium build quality"],
                    cons=["High price", "No headphone jack"]
                ),
                ProductReview(
                    rating=4,
                    comment="Great overall phone, but a bit expensive for what it offers.",
                    pros=["Beautiful screen", "Fast performance", "Good battery"],
                    cons=["Expensive", "Camera could be better in low light", "Limited storage options"]
                ),
                ProductReview(
                    rating=4,
                    comment="Solid premium smartphone that competes well with other flagships.",
                    pros=["Powerful processor", "Elegant design", "Great multitasking"],
                    cons=["No expandable storage", "Average low-light photography"]
                )
            ],
            average_rating=4.33,
            recommendation="Highly Recommended for those who value premium features and don't mind the price."
        )
    else:
        return ProductReviewResult(
            product_name="Generic Product",
            reviews=[
                ProductReview(
                    rating=3,
                    comment="Average product, nothing special.",
                    pros=["Works as expected", "Good value"],
                    cons=["Basic features only", "Average build quality"]
                )
            ],
            average_rating=3.0,
            recommendation="Recommended for budget-conscious buyers."
        )


@task
async def analyze_reviews(*, agent: Agent, product_info: str) -> ProductReviewResult:
    """Analyze product reviews and return a structured result.
    
    In a real implementation, this would call an LLM through the agent.
    For this example, we use a mock implementation.
    """
    print(f"Generating product analysis for product info...")
    
    # In a real implementation, you would use the agent to call the LLM:
    # return await agent(
    #     messages=[...],
    #     model="your-model",
    #     task_name="analyze_reviews",
    #     response_model=ProductReviewResult
    # )
    
    # For this example, we use a mock implementation
    return mock_review_analysis(product_info)


@flow
async def review_analysis_flow(*, flow_arguments: ValidationFlowArguments, agent: Agent) -> None:
    """Flow for analyzing product reviews with validation."""
    await analyze_reviews(agent=agent, product_info=flow_arguments.product_description)


# Deliberately trigger validation errors to demonstrate how they're handled
async def demonstrate_validation() -> None:
    """Show validation in action with valid and invalid data."""
    print("\n--- Valid Data Example ---")
    valid_data = {
        "product_name": "Smartphone X",
        "reviews": [
            {
                "rating": 5,
                "comment": "Great product!",
                "pros": ["Fast", "Good camera"],
                "cons": ["Expensive"]
            }
        ],
        "average_rating": 4.5,
        "recommendation": "Recommended"
    }
    
    try:
        # Parse valid data
        valid_result = ProductReviewResult.model_validate(valid_data)
        print("Validation passed successfully.")
        print(f"Product: {valid_result.product_name}")
        print(f"First review rating: {valid_result.reviews[0].rating}")
    except ValidationError as e:
        print("UNEXPECTED: Valid data failed validation!")
    
    print("\n--- Invalid Data Example ---")
    invalid_data = {
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
        # Try to parse invalid data (will fail)
        ProductReviewResult.model_validate(invalid_data)
        print("UNEXPECTED: Invalid data passed validation!")
    except ValidationError as e:
        print("Validation error caught:")
        for error in e.errors():
            print(f"- {error['loc']}: {error['msg']}")
        
        print("\nThis demonstrates how Goose uses Pydantic's validation to ensure")
        print("all LLM responses conform to the expected schema.")


async def main():
    """Run the review analysis flow and demonstrate validation."""
    # Create a unique run ID
    run_id = f"reviews-{os.getpid()}"
    
    print("=== Result Validation Example ===")
    print("This example demonstrates how Goose validates model outputs")
    print("against expected schemas and handles validation errors.\n")
    
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
    async with review_analysis_flow.start_run(run_id=run_id) as run:
        try:
            # Generate the review analysis
            await review_analysis_flow.generate(ValidationFlowArguments(product_description=product_info))
            
            # Display the validated result
            review_result = run.get_result(task=analyze_reviews)
            print("\n--- Validated Result ---")
            print("=" * 50)
            print(f"Product: {review_result.product_name}")
            print(f"Average Rating: {review_result.average_rating:.2f}/5")
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
            print("DEMONSTRATING VALIDATION:")
            await demonstrate_validation()
            
        except Exception as e:
            print(f"Error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())