"""
Command-line interface for TheNexus cognitive engine.

Provides an interactive way to query the AI reasoning system.
"""

from typing import NoReturn
import argparse
import logging
import sys
from thenexus.core_engine import CognitiveCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interact with TheNexus SuperAI Core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --query "What is the nature of consciousness?"
  %(prog)s --query "Explain quantum entanglement"
        """
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Input query to process through the cognitive engine"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    # Validate query is not None or empty
    if not args.query:
        logger.error("Query cannot be empty")
        print("Error: Query cannot be empty", file=sys.stderr)
        sys.exit(1)

    # Validate query length
    if len(args.query) > 10000:
        logger.error(f"Query too long: {len(args.query)} characters")
        print(
            f"Error: Query exceeds maximum length of 10000 characters",
            file=sys.stderr
        )
        sys.exit(1)

    try:
        logger.info(f"Processing query: {args.query[:50]}...")

        # Initialize cognitive engine
        engine = CognitiveCore()

        # Process query
        result = engine.think(args.query)

        # Output result
        print(f"\nAI Response: {result}")

        logger.info("Query processed successfully")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        print(
            f"Error: Configuration file not found. "
            f"Please ensure config.yaml exists.",
            file=sys.stderr
        )
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
