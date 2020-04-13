from experiments import *



def run_all():
    print("Running all VI/PI Solvers")
    run_frozenlake_solvers()
    run_hunterschoice_solvers()

    print("Using Solvers to generate Policies")
    simulate_all_frozenlake_policies()
    simulate_all_hunterschoice_policies()

    print("Resetting QLearner Data")
    reset_qlearner_folders()

    print("Running All QLearner Experiments")
    run_all_qlearners()
    
    print("Running Final Evaluations")
    run_all_final_evaluations()

if __name__ == "__main__":
    run_all()
    
