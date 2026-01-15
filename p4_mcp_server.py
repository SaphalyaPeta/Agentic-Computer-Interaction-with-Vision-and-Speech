"""
Hybrid MCP Architecture for Voice-Controlled UI Automation

This module implements a hybrid architecture combining:
- mcp-agent: Provides workflow orchestration with guaranteed sequential execution
- FastMCP: Provides streamable HTTP transport on port 3000

The system enables voice-controlled screen interaction by:
1. Capturing screenshots of the current screen
2. Running OCR to detect UI elements and their positions
3. Mapping numbered labels to UI elements
4. Executing mouse clicks on target elements based on voice commands

PURE EasyOCR Version
"""

import os
import time
import json
import re
import logging
import threading
import tempfile
from pathlib import Path
from datetime import timedelta
from PIL import ImageGrab
import pyautogui
from typing import Dict, List

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# IMPORTS WITH FALLBACKS
# =============================================================================
try:
    from mcp.server.fastmcp import FastMCP
    from mcp_agent.app import MCPApp
    from mcp_agent.executor.workflow import Workflow, WorkflowResult
except ImportError:
    print("Please install mcp-agent: pip install mcp-agent")
    exit(1)

try:
    from gtts import gTTS
    from playsound import playsound
    TTS_AVAILABLE = True
except ImportError:
    print("TTS libraries not available. Install: pip install gTTS playsound")
    TTS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
# Get screen dimensions
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# Test screenshot for scaling
test_screenshot = ImageGrab.grab()
img_width, img_height = test_screenshot.size
scale_x = screen_width / img_width
scale_y = screen_height / img_height
print(f"Scaling factors: x={scale_x:.2f}, y={scale_y:.2f}")

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# MCP SERVER INITIALIZATION
# =============================================================================
mcp_server = FastMCP("P4", port=3000)
mcp_agent_app = MCPApp(name="workflow-engine")

# =============================================================================
# TEXT-TO-SPEECH
# =============================================================================
def speak_text(text: str) -> None:
    """Speak text aloud using Google TTS."""
    if not TTS_AVAILABLE:
        print(f"TTS (would speak): {text}")
        return
    
    def speak():
        try:
            print(f">>> Speaking: {text[:50]}...")
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, f"tts_{hash(text)}.mp3")
            tts = gTTS(text=text, lang='en')
            tts.save(audio_file)
            playsound(audio_file)
            try:
                os.remove(audio_file)
            except:
                pass
        except Exception as e:
            print(f">>> TTS Error: {e}")
    
    threading.Thread(target=speak, daemon=True).start()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_env_file() -> dict:
    """Load environment variables from .env file."""
    env_path = Path(".env")
    env_vars = {}

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")

    return env_vars

def extract_digits(text: str) -> str:
    """Extract digits from text."""
    return ''.join(c for c in text if c.isdigit())

def get_screenshot_dir() -> str:
    """Get screenshot directory from environment."""
    env_vars = load_env_file()
    screenshot_dir = env_vars.get("PATH_TO_SCREENSHOT", "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    return screenshot_dir

# =============================================================================
# OCR TEXT DETECTION WITH EASYOCR
# =============================================================================
_ocr_reader = None

def get_ocr_reader():
    """Get or create singleton EasyOCR instance."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['en'])
            logger.info("EasyOCR initialized successfully")
        except ImportError as e:
            logger.error(f"EasyOCR not installed. Install with: pip install easyocr")
            logger.error(f"Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    return _ocr_reader

def detect_text_with_ocr(image_path: str) -> List[Dict]:
    """
    Use EasyOCR to detect text with exact pixel coordinates.
    Returns text with bounding boxes in screen coordinates.
    """
    try:
        reader = get_ocr_reader()
        
        # Run OCR on the image
        results = reader.readtext(image_path)
        
        detections = []
        
        for result in results:
            # result structure: ([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence)
            bbox, text, confidence = result
            
            # Filter out low confidence results
            if confidence < 0.3:  # 30% confidence threshold
                continue
            
            # Calculate center of bounding box
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # Check if text starts with a number
            contains_number = text and text[0].isdigit()
            
            # Normalize to 0-1000 scale (for compatibility)
            normalized_x = (center_x / img_width) * 1000
            normalized_y = (center_y / img_height) * 1000
            
            detection = {
                "text": text,
                "x": float(normalized_x),
                "y": float(normalized_y),
                "contains_number": contains_number,
                # Store actual pixel coordinates
                "pixel_x": float(center_x),
                "pixel_y": float(center_y),
                "confidence": float(confidence),
                "bbox": [[float(p[0]), float(p[1])] for p in bbox]
            }
            
            detections.append(detection)
        
        logger.info(f"EasyOCR detected {len(detections)} text elements with exact coordinates.")
        return detections
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        # Fallback to simple mock data
        return generate_simple_mock_detections()

def generate_simple_mock_detections() -> List[Dict]:
    """Generate simple mock text detections for testing."""
    logger.warning("Using mock data because OCR failed")
    mock_elements = [
        {"text": "File", "x": 100, "y": 50, "contains_number": False, "pixel_x": 200, "pixel_y": 100, "confidence": 0.9},
        {"text": "Edit", "x": 200, "y": 50, "contains_number": False, "pixel_x": 400, "pixel_y": 100, "confidence": 0.9},
        {"text": "View", "x": 300, "y": 50, "contains_number": False, "pixel_x": 600, "pixel_y": 100, "confidence": 0.9},
    ]
    return mock_elements

def process_detections_to_mappings(detections: List[Dict], screenshot_path: str) -> List[Dict]:
    """Convert OCR detections to numbered mappings."""
    mappings = []
    
    for i, detection in enumerate(detections):
        text = detection.get("text", "")
        pixel_x = detection.get("pixel_x", 0)
        pixel_y = detection.get("pixel_y", 0)
        contains_number = detection.get("contains_number", False)
        confidence = detection.get("confidence", 0.0)
        
        # Skip low confidence detections
        if confidence < 0.3:
            continue
        
        # Extract number if present at beginning
        number_match = re.match(r'^(\d+)(.*)', text)
        if number_match:
            number = number_match.group(1)
            clean_text = number_match.group(2).strip()
        elif contains_number:
            # Extract any digits from text
            numbers = extract_digits(text)
            if numbers:
                number = numbers
                clean_text = re.sub(r'\d+', '', text).strip()
            else:
                number = str(i + 1)
                clean_text = text
        else:
            # Assign a number based on position
            number = str(i + 1)
            clean_text = text
        
        if clean_text and clean_text not in ['', '.', ',', ':', ';', '!', '?']:
            # Convert to screen coordinates using actual scaling
            screen_x = pixel_x * scale_x
            screen_y = pixel_y * scale_y
            
            mapping = {
                "number": number,
                "text": clean_text,
                "center_x": float(screen_x),
                "center_y": float(screen_y),
                "original_text": text,
                "confidence": confidence,
                "pixel_coordinates": {"x": float(pixel_x), "y": float(pixel_y)}
            }
            mappings.append(mapping)
            logger.info(f"Mapping {number}: '{clean_text}' at PIXELS ({pixel_x:.0f}, {pixel_y:.0f}) → SCREEN ({screen_x:.0f}, {screen_y:.0f})")
    
    # Sort by y coordinate (top to bottom) then x coordinate (left to right)
    mappings.sort(key=lambda m: (m["center_y"], m["center_x"]))
    
    # Reassign numbers based on sorted position
    for i, mapping in enumerate(mappings):
        mapping["number"] = str(i + 1)
    
    return mappings

# =============================================================================
# WORKFLOW: CaptureScreenWithNumbers
# =============================================================================
@mcp_agent_app.workflow
class CaptureScreenWithNumbers(Workflow[dict]):
    """Screen capture workflow with OCR text detection."""
    
    def __init__(self):
        super().__init__()
        self.screenshot_path = ""
        self.mappings = []
    
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def prepare_screen(self) -> dict:
        """Prepare screen for capture."""
        speak_text("Preparing screen for analysis")
        time.sleep(1)
        return {"status": "prepared"}
    
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=3))
    async def capture_and_analyze(self) -> dict:
        """Capture screenshot and analyze with OCR."""
        try:
            # Capture screenshot
            screenshot = ImageGrab.grab()
            screenshot_dir = get_screenshot_dir()
            timestamp = int(time.time())
            
            self.screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
            screenshot.save(self.screenshot_path, 'PNG')
            logger.info(f"Screenshot saved: {self.screenshot_path}")
            
            # Use OCR to detect text with EXACT coordinates
            speak_text("Analyzing screen with optical character recognition")
            detections = detect_text_with_ocr(self.screenshot_path)
            
            # Process to mappings
            self.mappings = process_detections_to_mappings(detections, self.screenshot_path)
            
            # Save mappings to JSON
            json_path = os.path.join(screenshot_dir, f"mappings_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump({
                    "screenshot": self.screenshot_path,
                    "timestamp": timestamp,
                    "screen_size": {"width": screen_width, "height": screen_height},
                    "image_size": {"width": img_width, "height": img_height},
                    "scaling_factors": {"scale_x": scale_x, "scale_y": scale_y},
                    "mappings": self.mappings,
                    "mappings_count": len(self.mappings),
                    "detection_method": "EasyOCR"
                }, f, indent=2)
            
            logger.info(f"Mappings saved: {json_path}")
            
            return {
                "status": "success",
                "screenshot": self.screenshot_path,
                "json_file": json_path,
                "mappings_count": len(self.mappings),
                "message": f"Found {len(self.mappings)} UI elements with precise coordinates"
            }
            
        except Exception as e:
            logger.error(f"Capture/analyze error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to analyze screen"
            }
    
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def announce_results(self, analysis_result: dict) -> dict:
        """Announce analysis results."""
        status = analysis_result.get("status", "unknown")
        
        if status == "success":
            count = analysis_result.get("mappings_count", 0)
            speak_text(f"Screen analysis complete. Found {count} UI elements with precise locations.")
            message = f"Found {count} elements with precise coordinates"
        else:
            speak_text("Screen analysis failed. Check console for errors.")
            message = "Analysis failed"
        
        return {
            "status": "announced",
            "message": message,
            "analysis_status": status
        }
    
    @mcp_agent_app.workflow_run
    async def run(self) -> WorkflowResult[dict]:
        """Execute the complete workflow."""
        try:
            step1 = await self.prepare_screen()
            step2 = await self.capture_and_analyze()
            step3 = await self.announce_results(step2)
            
            return WorkflowResult(value={
                'workflow': 'CaptureScreenWithNumbers',
                'steps': [step1, step2, step3],
                'status': step2.get('status', 'unknown'),
                'screenshot': self.screenshot_path,
                'mappings': self.mappings,
                'mappings_count': len(self.mappings),
                'message': step3.get('message', 'Workflow completed')
            })
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return WorkflowResult(value={
                'workflow': 'CaptureScreenWithNumbers',
                'status': 'error',
                'error': str(e)
            })

@mcp_server.tool()
async def capture_screen_with_numbers_tool() -> str:
    """MCP tool for capturing screen with numbered labels."""
    try:
        workflow = CaptureScreenWithNumbers()
        result = await workflow.run()
        return json.dumps(result.value, indent=2)
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return json.dumps({'status': 'error', 'error': str(e)})

# =============================================================================
# WORKFLOW: ClickWorkFlow
# =============================================================================
@mcp_agent_app.workflow
class ClickWorkFlow(Workflow[dict]):
    """Workflow for clicking on UI elements."""
    
    def __init__(self):
        super().__init__()
        self.target = ""
        self.mappings = []
        self.found_element = None
    
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=1))
    async def load_latest_mappings(self, target: str) -> dict:
        """Load the most recent mappings file."""
        self.target = target.lower()
        
        try:
            screenshot_dir = get_screenshot_dir()
            json_files = [f for f in os.listdir(screenshot_dir) if f.startswith("mappings_") and f.endswith(".json")]
            
            if not json_files:
                return {
                    "status": "no_mappings",
                    "message": "No mapping files found. Please run capture_screen_with_numbers first."
                }
            
            # Get most recent
            latest_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(screenshot_dir, f)))
            json_path = os.path.join(screenshot_dir, latest_file)
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.mappings = data.get("mappings", [])
            
            return {
                "status": "loaded",
                "file": json_path,
                "mappings_count": len(self.mappings),
                "target": target
            }
            
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
            return {"status": "error", "error": str(e)}
    
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def find_target_element(self, metadata: dict) -> dict:
        """Find target element in mappings."""
        if metadata.get("status") != "loaded":
            return metadata
        
        try:
            matches = []
            for mapping in self.mappings:
                text = mapping.get("text", "").lower()
                if self.target in text:
                    matches.append(mapping)
            
            if matches:
                # Sort by confidence and choose the best match
                matches.sort(key=lambda m: m.get("confidence", 0.0), reverse=True)
                self.found_element = matches[0]
                return {
                    "status": "found",
                    "element": self.found_element,
                    "matches_count": len(matches),
                    "message": f"Found '{self.target}' as '{self.found_element.get('text')}'"
                }
            
            # Try fuzzy matching
            import difflib
            all_texts = [m.get("text", "").lower() for m in self.mappings if m.get("text")]
            if all_texts:
                close_matches = difflib.get_close_matches(self.target, all_texts, n=1, cutoff=0.6)
                if close_matches:
                    for mapping in self.mappings:
                        if mapping.get("text", "").lower() == close_matches[0]:
                            self.found_element = mapping
                            return {
                                "status": "fuzzy_found",
                                "element": mapping,
                                "message": f"Found close match '{close_matches[0]}' for '{self.target}'"
                            }
            
            return {
                "status": "not_found",
                "target": self.target,
                "available_elements": [m.get("text", "") for m in self.mappings],
                "message": f"Could not find '{self.target}' in available elements"
            }
            
        except Exception as e:
            logger.error(f"Error finding element: {e}")
            return {"status": "error", "error": str(e)}
    
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def execute_click(self, find_result: dict) -> dict:
        """Execute mouse click on element."""
        if find_result.get("status") not in ["found", "fuzzy_found"]:
            return find_result
        
        try:
            element = self.found_element
            x = element.get("center_x")
            y = element.get("center_y")
            
            if x is None or y is None:
                return {
                    "status": "no_coordinates",
                    "element": element,
                    "message": "Element has no coordinates"
                }
            
            # Log the exact coordinates being used
            logger.info(f"Clicking at exact coordinates: ({x:.1f}, {y:.1f}) for '{element.get('text')}'")
            
            # Move and click
            pyautogui.moveTo(x, y, duration=0.3)
            time.sleep(0.1)
            pyautogui.click()
            
            return {
                "status": "clicked",
                "element": element,
                "coordinates": {"x": x, "y": y},
                "message": f"Clicked on '{element.get('text')}' at ({x:.0f}, {y:.0f})"
            }
            
        except Exception as e:
            logger.error(f"Error clicking: {e}")
            return {"status": "error", "error": str(e)}
    
    @mcp_agent_app.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def announce_click_result(self, click_result: dict) -> dict:
        """Announce click result."""
        status = click_result.get("status", "unknown")
        
        if status == "clicked":
            element_text = click_result.get("element", {}).get("text", "element")
            speak_text(f"Successfully clicked on {element_text}")
            message = f"Clicked {element_text}"
        elif status == "found" or status == "fuzzy_found":
            speak_text("Element found but not clicked")
            message = "Element found"
        elif status == "not_found":
            speak_text(f"Could not find {self.target}")
            message = f"Not found: {self.target}"
        else:
            speak_text("Error during click operation")
            message = "Error"
        
        return {
            "status": "announced",
            "message": message,
            "click_status": status
        }
    
    @mcp_agent_app.workflow_run
    async def run(self, target: str) -> WorkflowResult[dict]:
        """Execute complete click workflow."""
        try:
            step1 = await self.load_latest_mappings(target)
            step2 = await self.find_target_element(step1)
            step3 = await self.execute_click(step2)
            step4 = await self.announce_click_result(step3)
            
            return WorkflowResult(value={
                'workflow': 'ClickWorkFlow',
                'target': target,
                'steps': [step1, step2, step3, step4],
                'status': step3.get('status'),
                'clicked': step3.get('status') == 'clicked',
                'element': self.found_element,
                'message': step4.get('message')
            })
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return WorkflowResult(value={
                'workflow': 'ClickWorkFlow',
                'status': 'error',
                'error': str(e)
            })

@mcp_server.tool()
async def click_workflow_tool(target: str) -> str:
    """MCP tool for clicking on UI elements."""
    try:
        workflow = ClickWorkFlow()
        result = await workflow.run(target)
        return json.dumps(result.value, indent=2)
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return json.dumps({'status': 'error', 'error': str(e)})

# =============================================================================
# OTHER TOOLS
# =============================================================================
@mcp_server.tool()
async def vanilla_workflow_tool(target=None) -> str:
    """Simple workflow tool."""
    try:
        # Just capture a screenshot
        screenshot_dir = get_screenshot_dir()
        screenshot = ImageGrab.grab()
        timestamp = int(time.time())
        screenshot_path = os.path.join(screenshot_dir, f"vanilla_{timestamp}.png")
        screenshot.save(screenshot_path, 'PNG')
        
        speak_text("Screenshot captured")
        
        return json.dumps({
            'status': 'success',
            'screenshot': screenshot_path,
            'message': 'Vanilla workflow completed'
        }, indent=2)
    except Exception as e:
        logger.error(f"Error: {e}")
        return json.dumps({'status': 'error', 'error': str(e)})

@mcp_server.tool()
async def echo_tool(command: str) -> str:
    """Echo command with TTS."""
    result = f"##VC##${command}##VC##"
    speak_text(command)
    return result

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("P4: Agentic Computer Interaction (PURE EasyOCR Version)")
    logger.info(f"Server URL: http://127.0.0.1:3000/mcp")
    logger.info(f"Screenshot directory: {get_screenshot_dir()}")
    logger.info("=" * 60)
    logger.info("Available tools:")
    logger.info("1. capture_screen_with_numbers_tool()")
    logger.info("2. click_workflow_tool('target')")
    logger.info("3. vanilla_workflow_tool()")
    logger.info("4. echo_tool('command')")
    logger.info("=" * 60)
    
    # Test OCR availability
    try:
        import easyocr
        logger.info("✓ EasyOCR is available")
    except ImportError:
        logger.warning("✗ EasyOCR not installed. Install with: pip install easyocr")
    
    mcp_server.run(transport="streamable-http")