"""Compile equation strings into programs for a numba stack machine.

Lexing and parsing are done by ast.parse. _Translator turns the tree into stack ops.
A program is a float64 array [n_coefs, n_ops, coef_0..coef_K-1, op_0, arg_0, op_1, arg_1, ...].
The x equation uses a0..aK and the y equation uses b0..bK, so step(px, py, a, b) keeps its signature.
"""

import ast
import re

import numba
import numpy as np

HEADER = 2  # slots holding n_coefs and n_ops at the start of a program

OP_PUSH_CONST, OP_PUSH_COEF, OP_PUSH_X, OP_PUSH_Y = 0, 1, 2, 3
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW = 4, 5, 6, 7, 8
OP_NEG, OP_ABS, OP_SIN, OP_COS = 9, 10, 11, 12

BINARY_OPS = {
    ast.Add: OP_ADD,
    ast.Sub: OP_SUB,
    ast.Mult: OP_MUL,
    ast.Div: OP_DIV,
    ast.Pow: OP_POW,
}
FUNCTIONS = {"abs": OP_ABS, "sin": OP_SIN, "cos": OP_COS}


class _Translator:
    """Walks the AST in post-order and records (op, arg) pairs."""

    def __init__(self, coef_prefix):
        self.coef_prefix = coef_prefix
        self.code = []
        self.n_coefs = 0

    def visit(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            self.code.append((OP_PUSH_CONST, float(node.value)))
        
        elif isinstance(node, ast.Name):
            self._visit_name(node.id)
        
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
            self.visit(node.left)
            self.visit(node.right)
            self.code.append((BINARY_OPS[type(node.op)], 0.0))
        
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.UAdd)
        ):
            self.visit(node.operand)
            if isinstance(node.op, ast.USub):
                self.code.append((OP_NEG, 0.0))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            self.visit(node.args[0])
            self.code.append((FUNCTIONS[node.func.id], 0.0))
        else:
            raise ValueError(f"unsupported expression: {ast.unparse(node)}")

    def _visit_name(self, name):
        if name == "x":
            self.code.append((OP_PUSH_X, 0.0))
        elif name == "y":
            self.code.append((OP_PUSH_Y, 0.0))
        elif re.fullmatch(f"{self.coef_prefix}[0-9]+", name):
            index = int(name[1:])
            self.n_coefs = max(self.n_coefs, index + 1)
            self.code.append((OP_PUSH_COEF, float(index)))
        else:
            raise ValueError(f"unknown name '{name}'")


def compile_expression(src, coef_prefix):
    """Compile one equation string into a program array."""
    if "=" in src:  # accepts "x_i = ..." as well as a bare expression
        src = src.split("=", 1)[1]
    tree = ast.parse(src.strip(), mode="eval")

    translator = _Translator(coef_prefix)
    translator.visit(tree.body)

    coefs = np.zeros(translator.n_coefs)
    pairs = np.array(translator.code, dtype=np.float64).ravel()
    header = np.array([translator.n_coefs, len(translator.code)], dtype=np.float64)
    return np.concatenate((header, coefs, pairs))


def compile_system(x_src, y_src):
    """Compile the x and y equations into the (a, b) arrays that step takes."""
    return compile_expression(x_src, "a"), compile_expression(y_src, "b")


def random_program_coefs(prog, lo=-3.0, hi=3.0):
    """Return a copy of prog with random values in its coefficient slots."""
    prog = prog.copy()
    n_coefs = int(prog[0])
    prog[HEADER : HEADER + n_coefs] = np.random.uniform(lo, hi, n_coefs)
    return prog


def program_coefs(prog):
    """Return only the coefficient values stored in a program array."""
    return prog[HEADER : HEADER + int(prog[0])].copy()


def set_program_coefs(prog, coefs):
    """Return a copy of prog with its coefficient slots set to coefs."""
    prog = prog.copy()
    prog[HEADER : HEADER + len(coefs)] = coefs
    return prog


# error_model="numpy" makes division by zero give inf/nan instead of raising
@numba.njit(fastmath=True, error_model="numpy")
def run_program(prog, px, py):
    '''Stack machine for a program array. Given a program and a point (px, py), 
    executes the program by pushing and popping values on a stack.    

    The program is a `float64` array `[n_coefs, n_ops, coef_0..coef_K-1, op_0, arg_0, op_1, arg_1, ...]`, 
    where `n_coefs` is the number of coefficients, n_ops is the number of operations,
    `coef_0..coef_K-1` are the coefficient values, and `op_i`, `arg_i` are the operation 
    and argument pairs for each operation.
    
    Returns the top of the stack after executing the program.
    '''
    n_coefs = int(prog[0])
    n_ops = int(prog[1])
    code = HEADER + n_coefs  # index where the (op, arg) pairs start
    stack = np.empty(n_ops)  # a program never needs more slots than it has ops
    sp = 0  # number of values on the stack

    for k in range(n_ops):
        op = int(prog[code + 2 * k])
        arg = prog[code + 2 * k + 1]

        if op == OP_PUSH_CONST:
            stack[sp] = arg
            sp += 1
        elif op == OP_PUSH_COEF:
            stack[sp] = prog[HEADER + int(arg)]
            sp += 1
        elif op == OP_PUSH_X:
            stack[sp] = px
            sp += 1
        elif op == OP_PUSH_Y:
            stack[sp] = py
            sp += 1
        elif op == OP_NEG:
            stack[sp - 1] = -stack[sp - 1]
        elif op == OP_ABS:
            stack[sp - 1] = abs(stack[sp - 1])
        elif op == OP_SIN:
            stack[sp - 1] = np.sin(stack[sp - 1])
        elif op == OP_COS:
            stack[sp - 1] = np.cos(stack[sp - 1])
        else:  # binary ops pop the right operand and combine it with the left one
            right = stack[sp - 1]
            left = stack[sp - 2]
            sp -= 1
            if op == OP_ADD:
                stack[sp - 1] = left + right
            elif op == OP_SUB:
                stack[sp - 1] = left - right
            elif op == OP_MUL:
                stack[sp - 1] = left * right
            elif op == OP_DIV:
                stack[sp - 1] = left / right
            else:
                stack[sp - 1] = left**right

    return stack[0]


@numba.njit(fastmath=True)
def program_step(px, py, a, b):
    return run_program(a, px, py), run_program(b, px, py)


if __name__ == "__main__":
    from config import CreateConfig, RenderConfig, BASE_DIR
    from equations import DynamicEquation, abs_step
    from lyapunov_exponents import compute_lyapunov, createAttractors
    from parallel import draw_attractors_in_parallel
    from colors import create_linear_colormap
    import glob

    # the same map as abs_step, written as strings, must give the same point
    a, b = compile_system(
        "a0 + a1*x + a2*y + abs(a3*x) + abs(a4*y)",
        "b0 + b1*x + b2*y + abs(b3*x) + abs(b4*y)",
    )
    a, b = random_program_coefs(a), random_program_coefs(b)
    same = np.allclose(
        program_step(0.3, -0.7, a, b),
        abs_step(0.3, -0.7, a[HEADER : HEADER + 5], b[HEADER : HEADER + 5]),
    )
    print("matches abs_step:", same)

    # # a string equation run through createAttractors, saves images and JSON files
    equation = DynamicEquation(
        "sine_map", 
        "a0 + a1*abs(x) + a2*y", 
        "b0 + b1*cos(y) + b2*x"
    )
    out = BASE_DIR / "tryout"
    create_cfg = CreateConfig(
        equation=equation,
        base_dir=out,
        examples=120,
        iterations=100_000,
    )
    render_cfg = RenderConfig(
        base_dir=out,
        iterations=100_000_000,
        interpolation="histogram",
        filter_threshold=0.02,
    )
    createAttractors(create_cfg)
    files = glob.glob(f"{create_cfg.out_path}/*.json")

    for cmap in ["binary", "gray", "twilight", "inferno", 'terrain']:
        draw_attractors_in_parallel(
            files,
            render_cfg,
            cmap=create_linear_colormap(preset=cmap),
            n_processes=12,
            batch_size=24,
        )
