import subprocess, os, json

def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode == 0

def main():
    phase2_tests = [
        "python firmware/audio/tests/test_dsp.py",
        "python firmware/audio/tests/test_agc.py",
        "python firmware/audio/tests/test_vad.py",
        "python firmware/audio/tests/test_vad_agc_order.py",
        "python firmware/audio/tests/test_noise_suppression.py",
        "python firmware/audio/tests/test_vad_ns_agc_order.py",
        "python firmware/audio/tests/test_aec.py",
        "python firmware/audio/tests/test_aec_dtd.py",
        "python firmware/audio/tests/test_aec_dtd_interface.py",
        "python firmware/audio/tests/test_audio_e2e.py",
        "python firmware/audio/tests/test_integrated_pipeline.py"
    ]
    phase3_tests = [
        "python firmware/hal/tests/test_hal_audio.py",
        "python firmware/hal/tests/test_esp32_audio.py",
        "python firmware/hal/tests/test_device_manager.py",
        "python firmware/hal/tests/test_imu.py",
        "python firmware/hal/tests/test_touch.py",
        "python firmware/hal/tests/test_camera.py",
        "python firmware/hal/tests/test_device_integration.py",
        "python firmware/hal/tests/test_phase3_e2e.py"
    ]
    phase4_tests = [
        "python backend/tests/test_vision_pipeline.py",
        "python backend/tests/test_vision_scheduler.py",
        "python backend/tests/test_phase4_e2e.py"
    ]
    phase5_tests = [
        "python backend/tests/test_ai_contracts.py",
        "python backend/tests/test_object_detection.py",
        "python backend/tests/test_ocr.py",
        "pytest backend/tests/test_tesseract_ocr.py",
        "pytest backend/tests/test_ocr_benchmark.py",
        "python backend/tests/test_scene_analysis.py",
        "python backend/tests/test_engine_registry.py",
        "python backend/tests/test_vision_orchestrator.py",
        "python backend/tests/test_phase5_e2e.py"
    ]
    
    p2_pass = sum(1 for t in phase2_tests if run(t))
    p3_pass = sum(1 for t in phase3_tests if run(t))
    p4_pass = sum(1 for t in phase4_tests if run(t))
    p5_pass = sum(1 for t in phase5_tests if run(t))
    
    all_tests = len(phase2_tests) + len(phase3_tests) + len(phase4_tests) + len(phase5_tests)
    passed = p2_pass + p3_pass + p4_pass + p5_pass
    
    res = {
        "phase": 5,
        "status": "COMPLETE" if passed == all_tests else "INCOMPLETE",
        "test_total": all_tests,
        "passed": passed,
        "failed": all_tests - passed,
        "skipped": 0,
        "phase2_total": len(phase2_tests),
        "phase2_passed": p2_pass,
        "phase2_failed": len(phase2_tests) - p2_pass,
        "phase3_total": len(phase3_tests),
        "phase3_passed": p3_pass,
        "phase3_failed": len(phase3_tests) - p3_pass,
        "e2e_status": "PASS" if p5_pass == len(phase5_tests) else "FAIL",
        "deterministic": True,
        "physical_hardware": "NOT VALIDATED",
        "esp_idf_status": "NOT VERIFIED"
    }
    
    os.makedirs('tests/results', exist_ok=True)
    with open('tests/results/phase5-ai-vision.json', 'w') as f:
        json.dump(res, f, indent=4)
        
    print(f"P2: {p2_pass}/{len(phase2_tests)} P3: {p3_pass}/{len(phase3_tests)} P4: {p4_pass}/{len(phase4_tests)} P5: {p5_pass}/{len(phase5_tests)}")
    if passed == all_tests:
        print("ALL TESTS PASS")

if __name__ == "__main__":
    main()
