import ast
import json
import math
import operator
from dataclasses import dataclass

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class ToolContext:
    store: object
    user: str
    session: str


class Registry:
    def __init__(self):
        self.tools = {}

    def register(self, name, description, schema, handler):
        if name in self.tools:
            raise ValueError(f"Duplicate tool: {name}")
        Draft202012Validator.check_schema(schema)
        self.tools[name] = (description, schema, handler)

    def schemas(self):
        return [{"type": "function", "function": {"name": name, "description": value[0],
                                                       "parameters": value[1]}}
                for name, value in self.tools.items()]

    def execute(self, name, raw_arguments, context):
        try:
            if name not in self.tools:
                raise ValueError("Unknown tool")
            if len(raw_arguments) > 8000:
                raise ValueError("Tool arguments exceed 8000 characters")
            arguments = json.loads(raw_arguments)
            _, schema, handler = self.tools[name]
            errors = list(Draft202012Validator(schema).iter_errors(arguments))
            if errors:
                # Do not echo arbitrary raw input in validation errors.
                raise ValueError("Invalid arguments at " + str(list(errors[0].path)) +
                                 " (" + str(errors[0].validator) + ")")
            return {"ok": True, "data": handler(context, **arguments)}
        except Exception as exc:
            if isinstance(exc, (ValueError, ArithmeticError, SyntaxError)):
                message = str(exc)[:300]
            else:
                message = "Tool execution failed; inspect the tool implementation."
            return {"ok": False, "error": {"type": type(exc).__name__, "message": message}}


def calculate(expression):
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 50:
        raise ValueError("Expression is too complex")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod, ast.Pow: operator.pow}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 12 or (left < 0 and right % 1)):
                raise ValueError("Exponent must be real with absolute value <= 12")
            value = binary[type(node.op)](left, right)
        else:
            raise ValueError("Only numeric arithmetic is allowed")
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Numeric result exceeds safe bounds")
        return value

    return {"value": visit(tree.body)}


SEARCH_DATA = [
    {"title": "厦门出差安排", "text": "周五前完成周报，出差前确认酒店和车票。"},
    {"title": "Agent 循环说明", "text": "模型选择工具，runtime 校验参数并执行，再把结果交回模型。"},
    {"title": "上下文与记忆", "text": "最近对话保留原文，早期对话压缩为摘要；待办独立存储。"},
]


def search(context, query):
    terms = query.lower().split()
    hits = [item for item in SEARCH_DATA if any(term in (item["title"] + item["text"]).lower() for term in terms)]
    return {"mock": True, "query": query, "results": hits,
            "notice": "本地固定示例资料，不是实时网页搜索；没有匹配时返回空列表。"}


def weather(context, city):
    fixtures = {"广州": (28, "多云"), "厦门": (26, "晴"), "北京": (18, "晴"), "上海": (23, "小雨")}
    if city not in fixtures:
        raise ValueError("Mock weather supports 广州/厦门/北京/上海 only")
    temperature, condition = fixtures[city]
    return {"mock": True, "city": city, "temperature_c": temperature, "condition": condition,
            "notice": "固定演示数据，非真实天气、非今日预报。"}


def build_registry():
    registry = Registry()

    def schema(properties, required):
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}

    registry.register("calculator", "Calculate numeric arithmetic accurately; no variables or code.",
                      schema({"expression": {"type": "string", "minLength": 1, "maxLength": 160}}, ["expression"]),
                      lambda context, expression: calculate(expression))
    registry.register("search", "Search a small local MOCK corpus. Not live web search.",
                      schema({"query": {"type": "string", "minLength": 1, "maxLength": 200}}, ["query"]), search)
    registry.register("todo", "Manage todos ONLY in the current user/session. add requires title; complete/delete require id. "
                      "Use list to find the correct id before modifying an ambiguous item.",
                      schema({"action": {"type": "string", "enum": ["add", "list", "complete", "delete"]},
                              "title": {"type": "string", "minLength": 1, "maxLength": 300},
                              "id": {"type": "integer", "minimum": 1}}, ["action"]),
                      lambda context, action, title=None, id=None: context.store.todo(
                          context.user, context.session, action, title, id))
    registry.register("weather", "Return MOCK weather for 广州/厦门/北京/上海. Always disclose simulation.",
                      schema({"city": {"type": "string", "minLength": 1, "maxLength": 40}}, ["city"]), weather)
    return registry
