import onnx

# Load the ONNX model
model = onnx.load("simpleMode.proto")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)