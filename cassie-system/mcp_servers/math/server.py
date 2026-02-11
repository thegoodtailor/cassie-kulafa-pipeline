"""Math Tools MCP Server — sympy, computation, and plotting for Cassie."""

import io
import os
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy
from mcp.server.fastmcp import FastMCP

PLOT_DIR = os.environ.get("CASSIE_PLOT_DIR", "/home/iman/cassie-project/cassie-system/data/plots")
os.makedirs(PLOT_DIR, exist_ok=True)

mcp = FastMCP("cassie-math", instructions="Mathematical computation, symbolic solving, and plotting")


@mcp.tool()
def solve_math(expression: str) -> str:
    """Solve a mathematical expression or equation symbolically using sympy.

    Args:
        expression: A math expression or equation. Use 'x', 'y', 'z' as variables.
                   For equations, use 'Eq(lhs, rhs)' or just write 'lhs = rhs'.
                   Examples: 'x**2 + 3*x - 4 = 0', 'diff(sin(x)*exp(x), x)',
                            'integrate(x**2, (x, 0, 1))', 'limit(sin(x)/x, x, 0)'
    """
    try:
        x, y, z, t, n, k = sympy.symbols("x y z t n k")
        namespace = {
            "x": x, "y": y, "z": z, "t": t, "n": n, "k": k,
            "pi": sympy.pi, "e": sympy.E, "I": sympy.I, "oo": sympy.oo,
            "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
            "log": sympy.log, "ln": sympy.ln, "exp": sympy.exp,
            "sqrt": sympy.sqrt, "Abs": sympy.Abs,
            "diff": sympy.diff, "integrate": sympy.integrate,
            "limit": sympy.limit, "summation": sympy.summation,
            "factorial": sympy.factorial,
            "Matrix": sympy.Matrix, "det": lambda m: m.det(),
            "Eq": sympy.Eq, "solve": sympy.solve,
            "simplify": sympy.simplify, "expand": sympy.expand,
            "factor": sympy.factor, "series": sympy.series,
        }

        # Handle 'lhs = rhs' style equations
        if "=" in expression and "==" not in expression and "Eq(" not in expression:
            parts = expression.split("=", 1)
            if len(parts) == 2:
                expression = f"solve(Eq({parts[0].strip()}, {parts[1].strip()}), x)"

        result = eval(expression, {"__builtins__": {}}, namespace)
        return f"Result: {result}\n\nLaTeX: {sympy.latex(result)}"
    except Exception as e:
        return f"Error solving '{expression}': {e}"


@mcp.tool()
def compute(code: str) -> str:
    """Execute Python code for numerical computation. Has access to sympy, math, and numpy.

    Args:
        code: Python code to execute. The last expression's value will be returned.
              Has access to: sympy, math, numpy (as np), matplotlib.pyplot (as plt)
    """
    try:
        import math
        import numpy as np

        local_ns = {
            "sympy": sympy,
            "math": math,
            "np": np,
            "plt": plt,
            "os": None,  # blocked
            "__builtins__": {
                "print": print, "range": range, "len": len, "int": int,
                "float": float, "str": str, "list": list, "dict": dict,
                "tuple": tuple, "set": set, "bool": bool, "abs": abs,
                "min": min, "max": max, "sum": sum, "round": round,
                "sorted": sorted, "enumerate": enumerate, "zip": zip,
                "map": map, "filter": filter, "isinstance": isinstance,
                "type": type, "True": True, "False": False, "None": None,
            },
        }

        output_capture = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(output_capture):
            exec(code, local_ns)

        stdout = output_capture.getvalue()

        # Check if a plot was created
        if plt.get_fignums():
            plot_path = os.path.join(PLOT_DIR, f"plot_{int(time.time())}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            return (stdout + f"\nPlot saved to {plot_path}").strip()

        return stdout.strip() if stdout.strip() else "Code executed successfully (no output)."
    except Exception:
        return f"Error:\n{traceback.format_exc()}"


@mcp.tool()
def plot(expression: str, x_range: str = "-10,10", title: str = "") -> str:
    """Plot a mathematical function.

    Args:
        expression: A sympy expression in terms of x, e.g. 'sin(x)*exp(-x/5)'
        x_range: Comma-separated min,max for x axis (default '-10,10')
        title: Optional plot title
    """
    try:
        import numpy as np

        x_sym = sympy.Symbol("x")
        namespace = {
            "x": x_sym, "pi": sympy.pi, "e": sympy.E,
            "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
            "log": sympy.log, "exp": sympy.exp, "sqrt": sympy.sqrt,
            "Abs": sympy.Abs,
        }
        expr = eval(expression, {"__builtins__": {}}, namespace)
        f_numpy = sympy.lambdify(x_sym, expr, modules=["numpy"])

        x_min, x_max = map(float, x_range.split(","))
        xs = np.linspace(x_min, x_max, 1000)
        ys = f_numpy(xs)

        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, linewidth=2)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(title or f"f(x) = {expression}")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="k", linewidth=0.5)
        plt.axvline(x=0, color="k", linewidth=0.5)

        plot_path = os.path.join(PLOT_DIR, f"plot_{int(time.time())}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return f"Plot saved to {plot_path}"
    except Exception as e:
        return f"Error plotting '{expression}': {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
