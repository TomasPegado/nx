# Elixir Nx Performance & Compilation Guide

## Introduction

In Elixir Nx, understanding the distinction between `defn`, backends, and compilers is essential for writing performant numerical code.

- **`defn`** is used to define numerical functions that can be traced and compiled.
- **Backends** determine where the tensors live and where each operation is executed (e.g., CPU, GPU).
- **Compilers** define how entire `defn` functions are compiled to a single, optimized binary — e.g., using EXLA or Torchx — instead of evaluating one operation at a time in Elixir.

Failing to explicitly set the compiler can result in inefficient evaluation via `Nx.Defn.Evaluator`, which incurs overhead for each operation.

---

## Benchmarking Strategies

### Quick Benchmarking with `:timer.tc`

Elixir's built-in `:timer.tc` is useful for quick measurements:

```elixir
{time_us, _result} = :timer.tc(fn -> MyModule.my_function(input) end)
IO.puts("Elapsed time: #{time_us / 1_000_000}s")
```

### More Robust Benchmarking with Benchee

For more reliable and repeated benchmarking:

```elixir
Benchee.run(%{
  "MyFunction" => fn -> MyModule.my_function(input) end
})
```

## Backends vs Compilers

### Backends

Backends define **where** tensors are allocated and where each operation is executed — for example, on the CPU or GPU.

Setting a backend ensures that individual operations (e.g., matrix multiplications) are executed on the selected device.

#### How to Set a Backend

You can set a global default backend like this:

```elixir
Nx.global_default_backend(EXLA.Backend)
```

This ensures that all tensors and operations default to running on EXLA.Backend (e.g., NVIDIA GPU via CUDA, or CPU depending on configuration).

- Important: Setting the backend alone does not optimize how entire numerical functions (defn) are compiled or executed.

### Compilers

Compilers determine **how** the entire numerical function defined with `defn` is compiled and executed.

By default, if no compiler is set, `Nx.Defn` uses the `Evaluator`, which interprets each operation individually in Elixir. While this is helpful for development and debugging, it introduces significant performance overhead for tasks like inference or training.

By explicitly setting a compiler such as `EXLA`, you enable the function to be compiled into a **unified, optimized binary**. This binary runs entirely on the target device (CPU or GPU), avoiding costly back-and-forth between Elixir and the execution backend.

#### How to Set the Compiler Globally

```elixir
Nx.Defn.default_options(compiler: EXLA)
```
