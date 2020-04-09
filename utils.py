def get_vipi_results(
    environment, discount_rates=[0.1, 0.5, 0.9, 0.99, 0.999], **kwargs
):
    results = {}
    for discount in discount_rates:
        print(f"Running VI and PI with discount rate: {discount}")

        solvervi = run_solver(environment, ValueIteration, discount, **kwargs)
        solverpi = run_solver(
            environment, PolicyIteration, discount, eval_type=1, **kwargs
        )

        results[f"vi_{discount}_solver"] = solvervi
        results[f"pi_{discount}_solver"] = solverpi

    return results


def run_solver(environment, solver_func, *solver_args, **solver_kwargs):
    T, R = environment.build_TR_matrices()
    solver = solver_func(T, R, *solver_args, **solver_kwargs)

    solver.run()

    return solver
