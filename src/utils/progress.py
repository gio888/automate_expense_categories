"""
Progress tracking and user feedback utilities
"""

import time
import sys
from typing import Optional, Callable, Any, Tuple

class ProgressBar:
    """Simple text-based progress bar"""
    
    def __init__(self, total: int, description: str = "", width: int = 50):
        self.total = total
        self.description = description
        self.width = width
        self.current = 0
        self.start_time = time.time()
        
    def update(self, increment: int = 1, message: str = ""):
        """Update progress bar"""
        self.current += increment
        if self.current > self.total:
            self.current = self.total
            
        # Calculate percentage
        percentage = (self.current / self.total) * 100
        filled_width = int(self.width * (self.current / self.total))
        
        # Create bar
        bar = '█' * filled_width + '░' * (self.width - filled_width)
        
        # Calculate time
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed_time / self.current) * (self.total - self.current)
        else:
            eta = 0
            
        # Format time
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{int(seconds // 60)}m {int(seconds % 60)}s"
            else:
                return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
        
        # Build output
        output = f"\r{self.description}: {bar} {percentage:5.1f}% "
        output += f"({self.current}/{self.total}) "
        output += f"[{format_time(elapsed_time)} < {format_time(eta)}]"
        
        if message:
            output += f" {message}"
            
        sys.stdout.write(output)
        sys.stdout.flush()
        
        if self.current >= self.total:
            print()  # New line when complete
            
    def finish(self, message: str = "Complete!"):
        """Mark progress as finished"""
        self.current = self.total
        self.update(0, message)

def with_progress(
    items, 
    description: str = "Processing", 
    show_progress: bool = True
) -> Any:
    """
    Wrap an iterable with progress tracking
    
    Args:
        items: Iterable to process
        description: Description for progress bar
        show_progress: Whether to show progress bar
        
    Yields:
        Items from the iterable with progress tracking
    """
    if not show_progress or not hasattr(items, '__len__'):
        # Just return items as-is if no progress needed
        yield from items
        return
        
    total = len(items)
    progress = ProgressBar(total, description)
    
    try:
        for i, item in enumerate(items):
            yield item
            progress.update(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        raise
    finally:
        if progress.current < progress.total:
            progress.finish("Interrupted")

class TaskProgress:
    """Higher-level task progress tracking"""
    
    def __init__(self, task_name: str, steps: int):
        self.task_name = task_name
        self.steps = steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        
    def start_step(self, step_name: str):
        """Start a new step"""
        self.current_step += 1
        step_start = time.time()
        
        if self.step_times:
            # Calculate time for previous step
            prev_duration = step_start - self.step_times[-1]
            print(f"   ✅ Previous step completed in {prev_duration:.1f}s")
        
        self.step_times.append(step_start)
        
        # Calculate overall progress
        progress_pct = (self.current_step - 1) / self.steps * 100
        elapsed = step_start - self.start_time
        
        print(f"\n📍 Step {self.current_step}/{self.steps}: {step_name}")
        print(f"⏱️  Progress: {progress_pct:.0f}% | Elapsed: {elapsed:.1f}s")
        
    def complete_task(self):
        """Mark entire task as complete"""
        total_time = time.time() - self.start_time
        if self.step_times:
            last_step_time = time.time() - self.step_times[-1]
            print(f"   ✅ Final step completed in {last_step_time:.1f}s")
        
        print(f"\n🎉 {self.task_name} Complete!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        
        # Show step breakdown
        if len(self.step_times) > 1:
            print("⏱️  Step breakdown:")
            for i in range(1, len(self.step_times)):
                step_time = self.step_times[i] - self.step_times[i-1]
                print(f"   Step {i}: {step_time:.1f}s")

def show_spinner(duration: float, message: str = "Processing..."):
    """Show a simple spinner for operations without measurable progress"""
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    start_time = time.time()
    i = 0
    
    try:
        while time.time() - start_time < duration:
            sys.stdout.write(f"\r{spinner_chars[i % len(spinner_chars)]} {message}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        
        sys.stdout.write(f"\r✅ {message} Complete!\n")
        sys.stdout.flush()
        
    except KeyboardInterrupt:
        sys.stdout.write(f"\r❌ {message} Interrupted\n")
        sys.stdout.flush()
        raise

def safe_execute(
    func: Callable, 
    *args, 
    task_name: str = "Operation",
    show_progress: bool = True,
    **kwargs
) -> Tuple[bool, Any]:
    """
    Execute a function with progress tracking and error handling
    
    Returns:
        Tuple of (success: bool, result: Any)
    """
    if show_progress:
        print(f"🚀 Starting: {task_name}")
        start_time = time.time()
    
    try:
        result = func(*args, **kwargs)
        
        if show_progress:
            elapsed = time.time() - start_time
            print(f"✅ {task_name} completed in {elapsed:.1f}s")
        
        return True, result
        
    except Exception as e:
        if show_progress:
            elapsed = time.time() - start_time
            print(f"❌ {task_name} failed after {elapsed:.1f}s")
            print(f"   Error: {str(e)}")
        
        return False, str(e)

# Convenience functions for common operations
def progress_map(func: Callable, items, description: str = "Processing items"):
    """Apply function to items with progress tracking"""
    results = []
    
    for item in with_progress(items, description):
        results.append(func(item))
    
    return results

if __name__ == "__main__":
    # Test progress utilities
    import random
    
    print("Testing progress utilities...")
    
    # Test basic progress bar
    items = list(range(20))
    results = []
    
    for item in with_progress(items, "Processing numbers"):
        # Simulate work
        time.sleep(0.1)
        results.append(item * 2)
    
    print(f"Results: {results[:5]}...")
    
    # Test task progress
    task = TaskProgress("Example Task", 3)
    
    task.start_step("Loading data")
    time.sleep(0.5)
    
    task.start_step("Processing data")
    time.sleep(0.3)
    
    task.start_step("Saving results")
    time.sleep(0.2)
    
    task.complete_task()
    
    # Test spinner
    show_spinner(1.0, "Loading models")
    
    print("✅ Progress utility tests complete!")