#!/usr/bin/env python3
"""
Claude Skill System V2 - With Actual Function Execution
Loads Python functions from skills and registers them as callable tools
"""

import os
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Optional, Callable


class Skill:
    """Represents a loaded skill with executable functions"""

    def __init__(self, name: str, description: str, path: Path):
        self.name = name
        self.description = description
        self.path = path
        self.modules = {}
        self.functions = {}  # Registered callable functions

    def load_module(self, module_name: str):
        """Dynamically load a Python module from the skill directory"""
        module_path = self.path / f"{module_name}.py"
        if not module_path.exists():
            raise FileNotFoundError(f"Module {module_name} not found in skill {self.name}")

        spec = importlib.util.spec_from_file_location(f"{self.name}.{module_name}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.modules[module_name] = module

        # Register all public functions from this module
        # Use simplified naming: just the function name
        # This makes it easier for Claude to call the right tool
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                # Simple function name for tools
                self.functions[name] = obj

        return module

    def get_module(self, module_name: str):
        """Get a loaded module or load it if not already loaded"""
        if module_name not in self.modules:
            return self.load_module(module_name)
        return self.modules[module_name]

    def load_all_modules(self):
        """Load all Python modules in the skill directory"""
        for py_file in self.path.glob("*.py"):
            if py_file.name != "__init__.py":
                self.load_module(py_file.stem)


class SkillManager:
    """Manages skills and provides executable tools"""

    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.skills = {}
        self.all_functions = {}  # All registered functions from all skills
        self._load_skills()

    def _load_skills(self):
        """Load all skills from the skills directory"""
        if not self.skills_dir.exists():
            return

        for skill_path in self.skills_dir.iterdir():
            if skill_path.is_dir() and (skill_path / "SKILL.md").exists():
                self._load_skill(skill_path)

    def _load_skill(self, skill_path: Path):
        """Load a single skill and all its modules"""
        skill_md = skill_path / "SKILL.md"
        with open(skill_md, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse skill metadata
        name = skill_path.name
        description = ""

        for line in content.split('\n'):
            if line.startswith('description'):
                description = line.split('\t', 1)[1] if '\t' in line else ""
                break

        skill = Skill(name, description, skill_path)

        # Load all Python modules from this skill
        skill.load_all_modules()

        # Register all functions globally
        self.all_functions.update(skill.functions)

        self.skills[name] = skill

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name"""
        return self.skills.get(name)

    def list_skills(self) -> list[str]:
        """List all available skills"""
        return list(self.skills.keys())

    def get_tools_for_anthropic(self, skill_names: list[str] = None) -> list[dict]:
        """
        Build Anthropic tool definitions from skill functions

        Returns:
            List of tool definitions for Anthropic API
        """
        if skill_names is None:
            skill_names = self.list_skills()

        tools = []

        for skill_name in skill_names:
            skill = self.get_skill(skill_name)
            if not skill:
                continue

            for func_name, func in skill.functions.items():
                # Extract function signature and docstring
                sig = inspect.signature(func)
                doc = inspect.getdoc(func) or f"Function from {skill_name}"
                
                # Add skill context to description
                full_doc = f"[Skill: {skill_name}]\n{doc}"

                # Build parameter schema
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

                for param_name, param in sig.parameters.items():
                    # Simple type inference
                    param_type = "string"  # default
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation in (int, float):
                            param_type = "number"
                        elif param.annotation == bool:
                            param_type = "boolean"

                    parameters["properties"][param_name] = {
                        "type": param_type,
                        "description": f"Parameter {param_name}"
                    }

                    if param.default == inspect.Parameter.empty:
                        parameters["required"].append(param_name)

                tool_def = {
                    "name": func_name,
                    "description": full_doc,
                    "input_schema": parameters
                }

                tools.append(tool_def)

        return tools

    def call_function(self, func_name: str, **kwargs) -> Any:
        """Execute a registered function by name"""
        if func_name not in self.all_functions:
            raise ValueError(f"Function {func_name} not found")

        return self.all_functions[func_name](**kwargs)

    def build_system_prompt(self, skill_names: list[str] = None) -> str:
        """Build system prompt with skill documentation"""
        if skill_names is None:
            skill_names = self.list_skills()

        prompt_parts = []

        for skill_name in skill_names:
            skill = self.get_skill(skill_name)
            if skill:
                skill_md = skill.path / "SKILL.md"
                with open(skill_md, 'r', encoding='utf-8') as f:
                    prompt_parts.append(f"\n# Skill: {skill.name}\n\n{f.read()}")

        return "\n".join(prompt_parts)


def create_skill_enhanced_prompt(base_prompt: str, skill_manager: SkillManager,
                                  skill_names: list[str] = None) -> str:
    """
    Create an enhanced prompt with skill context and tool availability
    """
    skill_context = skill_manager.build_system_prompt(skill_names)

    enhanced_prompt = f"""{skill_context}

You have access to the skills above. You can call their functions to help answer questions.

---

User Question: {base_prompt}
"""

    return enhanced_prompt
