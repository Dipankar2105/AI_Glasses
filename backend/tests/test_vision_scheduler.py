import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.vision.scheduler import VisionScheduler

def test_scheduler():
    sched = VisionScheduler(max_fps=1.0)
    sched.push_frame("F1", 0.0)
    
    f, skip = sched.get_next_frame(0.0)
    assert f == "F1" and not skip
    
    sched.push_frame("F2", 0.5)
    f, skip = sched.get_next_frame(0.5)
    assert f == "F2" and skip # Throttled
    
    sched.push_frame("F3", 1.0)
    sched.push_frame("F4", 1.1)
    f, skip = sched.get_next_frame(1.1)
    assert f == "F4" and not skip # Latest frame behavior (F3 dropped implicitly)

if __name__ == "__main__":
    test_scheduler()
    print("test_vision_scheduler PASS")
