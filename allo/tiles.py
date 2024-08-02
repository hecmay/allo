import ast
import inspect
import astor


def generate_loops(op, args):
    # affine.for %arg3 = 0 to 64 {
    #   affine.for %arg4 = 0 to 64 {
    #     affine.for %arg5 = 0 to 64 {
    #       %0 = affine.load %arg1[%arg3, %arg5] : memref<64x64xf32>
    #       %1 = affine.load %arg2[%arg5, %arg4] : memref<64x64xf32>
    #       %2 = arith.mulf %0, %1 : f32
    #       %3 = affine.load %arg0[%arg3, %arg4] : memref<64x64xf32>
    #       %4 = arith.addf %3, %2 : f32
    #       affine.store %4, %arg0[%arg3, %arg4] : memref<64x64xf32>
    #     }
    #   }
    # }
    # build netsed loop from inner to outer
    code = []
    if op == "MatMult" or op == "BatchMatMult":
        indent = (len(args) + 1) * 2 * " "
        code.append(indent + f"%0 = affine.load %arg1[%arg3, %arg5] : memref<{args[0]}x{args[2]}xf32>")
        code.append(indent + f"%1 = affine.load %arg2[%arg5, %arg4] : memref<{args[2]}x{args[1]}xf32>")
        code.append(indent + f"%2 = arith.mulf %0, %1 : f32")
        code.append(indent + f"%3 = affine.load %arg0[%arg3, %arg4] : memref<{args[0]}x{args[1]}xf32>")
        code.append(indent + f"%4 = arith.addf %3, %2 : f32")
        code.append(indent + f"affine.store %4, %arg0[%arg3, %arg4] : memref<{args[0]}x{args[1]}xf32>")

        # from innner (last) to outer (first)
        for i in range(len(args) - 1, -1, -1):
            indent = (i + 2) * 2 * " "
            code = [ indent + f"affine.for %arg{i + 3} = 0 to {args[i]} {{" ] + code
            code.append(indent + "}")

    else:
        raise ValueError(f"Unsupported operation: {op}")
    
    return "\n".join(code)
        

class TileToLoops(ast.NodeTransformer):

    def __init__(self):
        self.arg_types = {}
        self.sram_buffers = {}
        self.tile_ops = []

    def visit_FunctionDef(self, node):
        assert len(self.arg_types) == 0

        for arg in node.args.args:
            if arg.annotation:
                ty = arg.annotation

                if isinstance(ty, ast.Subscript):
                    # Extract shape
                    shape = []
                    for r in ty.slice.elts:
                        if isinstance(r, ast.Name):
                            shape.append(r.id)
                        else:
                            raise ValueError(
                                f"Unsupported shape type: {type(r)}")

                    self.arg_types[arg.arg] = (ty.value.id, shape)

                else:
                    assert isinstance(ty, ast.Name)
                    self.arg_types[arg.arg] = (ty.id, [])

                # arg.annotation = None

        # Traverse function body and update
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def generic_visit(self, node):
        # Default behavior: visit all child nodes
        return super().generic_visit(node)

    def visit_AnnAssign(self, node):
        # print("===", type(node), astor.to_source(node))

        var_name = node.target.id
        assert var_name not in self.sram_buffers

        dtype = node.annotation.value.id
        shape = []

        assert isinstance(node.annotation.slice, ast.Tuple)
        ranges = node.annotation.slice.elts

        # Map from pointers to local variables
        for r in ranges:
            v = None
            if isinstance(r, ast.Name):
                v = r.id
            else:
                raise
            shape.append(v)

        self.sram_buffers[var_name] = (dtype, shape)

        # Check the right-hand side value
        if isinstance(node.value, ast.Subscript):
            self.tile_ops.append(["LOAD", node.target, node.value])
        else:
            assert isinstance(node.value, ast.BinOp)
            self.tile_ops.append(
                [node.value.op, node.value.left, node.value.right])

        assign_node = ast.Assign(
            targets=[node.target],
            value=node.value,
            type_comment=None
        )
        return assign_node

    def visit_Assign(self, node):
        # Assigning to existing buffers, e.g., sram or off-chip memory
        assert len(node.targets) == 1
        target = node.targets[0]

        assert isinstance(target, ast.Subscript)
        assert self.sram_buffers.get(target.value.id) or self.arg_types.get(
            target.value.id)

        self.tile_ops.append(["STORE", target, node.value])

        return node

    def emit(self, fn_name):

        indent = 2
        code = "module {" + f"\n" + indent * " " 
        code += f"func.func @{fn_name}("

        # Get grid dimensions and fn arguments
        grid_dim = 0
        index = 0
        for arg, ty in self.arg_types.items():
            dtype, shape = ty
            if len(shape) == 0:
                grid_dim += 1
            else:
                ty_str = 'x'.join(shape) + f"x{dtype}"
                code += f"%{arg}: memref<{ty_str}>"
                code += ", "
            index += 1
        code = code[:-2] # remove trailing comma
        code += ") {" + f"\n"

        # Generate the tile loops
        for op in self.tile_ops:
            if not isinstance(op[0], str):
                op_name = op[0].__class__.__name__
                if op_name == "MatMult":
                    # Matrix multiplication
                    op0 = op[1].id
                    op1 = op[2].id
                    assert self.sram_buffers.get(op0)
                    assert self.sram_buffers.get(op1)

                    # generate code for matrix multiplication
                    dim0 = self.sram_buffers[op0][1]
                    dim1 = self.sram_buffers[op1][1]
                    assert dim0[-1] == dim1[0]

                    code += generate_loops(op_name, [dim0[0], dim1[1], dim0[1]])

                else:
                    raise ValueError(f"Unsupported operation: {op[0]}")
            
            else:
                assert op[0] in ["LOAD", "STORE"]

 
        code += "\n" + indent * " " + "}" + f"\n"
        code += "}" + f"\n"
        return code


def as_tile(**kwargs):

    def decorator(f):
        source = inspect.getsource(f)
        tree = ast.parse(source)

        # Remove the decorators from the AST, because the modified function will
        # be passed to them anyway and we don't want them to be called twice.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.decorator_list.clear()

        # Make modifications to the AST here
        mutator = TileToLoops()
        tree = mutator.visit(tree)
        ast.fix_missing_locations(tree)

        # print(f"\n=== revised ===\n{astor.to_source(tree)}")
        name = f.__code__.co_name

        # code = compile(tree, name, 'exec')
        # temp_globals = dict(globals())
        # exec(code, temp_globals)
        # mod = temp_globals[name]

        mod = f 
        # mod.source = astor.to_source(tree)
        # print(mod.source)

        # temporary hack: ast.fix_missing_locations has a bug that causes the
        #   `inspect` extracted code string to be wrong, so we save str directly

        # Get the grid index arguments and build a for-loop in python
        mod.source = mutator.emit(f.__code__.co_name)  

        return mod

    return decorator
