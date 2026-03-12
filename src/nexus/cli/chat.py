"""
Nexus CLI - Daily AI interaction interface.

Usage:
    python -m nexus.cli.chat              # Interactive chat
    python -m nexus.cli.chat --model X    # Use specific model
    python -m nexus.cli.chat --status     # Check system status
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure proper encoding on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def print_banner():
    """Print welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                     NEXUS AI                                ║
║              Local-First Intelligence Platform                 ║
╚═══════════════════════════════════════════════════════════════╝
""")


def print_help():
    """Print help commands."""
    print("""
Commands:
  /help     - Show this help
  /status   - Check backend status
  /stats    - Show usage statistics
  /model X  - Switch to model preset X
  /clear    - Clear conversation
  /think    - Enable/disable thinking mode (Qwen3)
  /exit     - Exit chat

Models:
  ollama-qwen3-30b   - Primary (MoE, efficient)
  ollama-qwen3-8b    - Fast (laptop-friendly)
  ollama-llama-70b   - Alternative
  anthropic-sonnet   - Claude (paid)
""")


async def check_status(llm):
    """Check and display backend status."""
    print("\n📊 Backend Status:")
    availability = await llm.check_backends()
    
    for backend, available in availability.items():
        if available:
            print(f"  ✓ {backend}")
        else:
            print(f"  ✗ {backend} (unavailable)")
    print()


def show_stats(llm):
    """Show usage statistics."""
    stats = llm.get_stats()
    print("\n📈 Session Statistics:")
    print(f"  Requests: {stats['requests']}")
    print(f"  Tokens generated: {stats['tokens_generated']:,}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']*100:.1f}%")
    print(f"  Fallback rate: {stats['fallback_rate']*100:.1f}%")
    print(f"  Estimated cost: ${stats['cost_usd']:.4f}")
    print()


async def chat_loop(model_preset: str = "ollama-qwen3-30b"):
    """Main chat interaction loop."""
    from nexus.core.llm_provider import NexusLLM, RoutingConfig
    
    print_banner()
    
    # Initialize LLM
    config = RoutingConfig()
    llm = NexusLLM(routing_config=config)
    
    # Check Ollama availability
    print("🔌 Checking backends...")
    await check_status(llm)
    
    current_model = model_preset
    conversation_history = []
    thinking_mode = True  # For Qwen3
    
    print(f"💬 Using: {current_model}")
    print("   Type /help for commands, /exit to quit\n")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                args = user_input.split()[1:] if len(user_input.split()) > 1 else []
                
                if cmd == "/exit" or cmd == "/quit":
                    print("\n👋 Goodbye!")
                    break
                    
                elif cmd == "/help":
                    print_help()
                    continue
                    
                elif cmd == "/status":
                    await check_status(llm)
                    continue
                    
                elif cmd == "/stats":
                    show_stats(llm)
                    continue
                    
                elif cmd == "/clear":
                    conversation_history.clear()
                    llm.clear_cache()
                    print("🧹 Conversation cleared")
                    continue
                    
                elif cmd == "/model":
                    if args:
                        current_model = args[0]
                        print(f"🔄 Switched to: {current_model}")
                    else:
                        print(f"📌 Current model: {current_model}")
                    continue
                    
                elif cmd == "/think":
                    thinking_mode = not thinking_mode
                    status = "enabled" if thinking_mode else "disabled"
                    print(f"🧠 Thinking mode: {status}")
                    continue
                    
                else:
                    print(f"❓ Unknown command: {cmd}")
                    continue
            
            # Build context from conversation history
            context = ""
            if conversation_history:
                recent = conversation_history[-5:]  # Last 5 exchanges
                context = "\n".join([
                    f"User: {h['user']}\nAssistant: {h['assistant']}"
                    for h in recent
                ])
                context = f"Previous conversation:\n{context}\n\n"
            
            # Generate response
            print("\n🤖 Nexus: ", end="", flush=True)
            
            try:
                response = await llm.generate(
                    prompt=f"{context}User: {user_input}",
                    task_type="conversation",
                    system_prompt="You are Nexus, a helpful AI assistant. Be concise and helpful.",
                    force_model=current_model,
                )
                
                content = response["content"]
                print(content)
                
                # Show metadata
                tokens = response.get("tokens_used", 0)
                duration = response.get("duration_seconds", 0)
                model = response.get("model", "unknown")
                
                print(f"\n   [{model} | {tokens} tokens | {duration:.1f}s]")
                
                # Add to history
                conversation_history.append({
                    "user": user_input,
                    "assistant": content,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("   Try /status to check backend availability")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


async def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nexus AI Chat")
    parser.add_argument(
        "--model", "-m",
        default="ollama-qwen3-30b",
        help="Model preset to use"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Just check status and exit"
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List available model presets"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable model presets:")
        presets = [
            ("ollama-qwen3-30b", "Qwen3 30B MoE - Best efficiency"),
            ("ollama-qwen3-8b", "Qwen3 8B - Fast, laptop-friendly"),
            ("ollama-qwen3-14b", "Qwen3 14B - Middle ground"),
            ("ollama-qwen3-32b", "Qwen3 32B Dense - High quality"),
            ("ollama-qwen3-30b-fast", "Qwen3 30B without thinking"),
            ("ollama-qwen3-8b-fast", "Qwen3 8B without thinking"),
            ("ollama-llama-8b", "Llama 3.1 8B"),
            ("ollama-llama-70b", "Llama 3.1 70B"),
            ("anthropic-sonnet", "Claude Sonnet (paid)"),
            ("anthropic-haiku", "Claude Haiku (paid)"),
            ("openai-gpt4o", "GPT-4o (paid)"),
        ]
        for preset, desc in presets:
            print(f"  {preset:25} - {desc}")
        return
    
    if args.status:
        from nexus.core.llm_provider import NexusLLM
        llm = NexusLLM()
        await check_status(llm)
        return
    
    await chat_loop(model_preset=args.model)


if __name__ == "__main__":
    asyncio.run(main())
