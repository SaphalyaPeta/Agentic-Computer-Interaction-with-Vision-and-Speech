# Agentic-Computer-Interaction-with-Vision-and-Speech

This brings together the concepts to create an agentic system that interacts with your computer through vision and speech.

The system enables voice-controlled screen interaction by capturing screenshots, running OCR to detect UI elements and their positions, mapping numbered labels to UI elements, and executing mouse clicks on target elements based on voice commands.

Core capability: "Click the Settings button" → System captures screen → OCR identifies numbered labels → System calculates coordinates → Executes click

This assignment uses a hybrid architecture combining mcp-agent for workflow orchestration with guaranteed sequential execution and FastMCP for streamable HTTP transport.

Objectives:
Build an agentic system that perceives and acts on computer interfaces
Use OCR (PaddleOCR) for UI element detection and localization
Implement MCP workflows with guaranteed sequential execution
Handle coordinate mapping from OCR detection to actual screen coordinates
Understand the perception-action loop in real-time desktop automation
 

System Architecture
FastMCP Server
mcp-agent Workflows
Open WebUI

Implementation of two workflows and their corresponding MCP tools:

1. CaptureScreenWithNumbers Workflow:
This workflow captures the screen and creates a mapping between numbered labels and UI element text. The workflow should:
Announce "start listening" via TTS and wait for the UI to display numbered labels (you need to enable the Voice Control app in Mac or Voice Access app in Windows)
Announce "show numbers" to trigger numbered label display on screen
Note that you may need to issue other speech commands (e.g., bring an app to the foreground) to get a clean screenshot 
You may also consider reducing the screen resolution so that a screenshot takes less memory, and the OCR/VLM can process faster
Capture a screenshot using PIL.ImageGrab
Run PaddleOCR (or any better OCR or VLM model) on the screenshot to detect all text elements
Create mappings between numbered labels and UI text (handle both embedded numbers like "1Settings" and adjacent number-text pairs)
Save mappings to a JSON file with structure: {"mappings": [{"number": "1", "text": "Settings", "center_x": 100, "center_y": 200}, ...]}
Announce "stop listening" when complete
Tool: capture_screen_with_numbers_tool() - Exposes the workflow as an MCP tool.

2. ClickWorkFlow Workflow:
This workflow finds and clicks on a target UI element. The workflow should:
Read the OCR metadata of the most recently captured screenshot from the JSON file created by CaptureScreenWithNumbers
Search for the target element by text matching (case-insensitive substring)
Convert OCR coordinates to screen coordinates using the scaling factors
Move the mouse to the target position using pyautogui.moveTo()
Execute the click using pyautogui.click()
Announce "clicked on [target]" via TTS
Tool: click_workflow_tool(target: str) - Takes a target name (e.g., "Settings", "File") and executes the click workflow.
