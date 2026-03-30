import sys
sys.path.insert(0, r"C:\dev\Nexus\Nexus\src")
import asyncio

async def test():
    from nexus.platform import NexusPlatform
    platform = NexusPlatform()
    status = await platform.initialize()
    
    print("\nPlatform Status:")
    for component, ok in status.items():
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon} {component}")
    
    return all(status.values())

result = asyncio.run(test())
print(f"\nOverall: {'SUCCESS' if result else 'PARTIAL'}")
