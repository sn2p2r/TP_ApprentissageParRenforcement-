import os

def run_script(path):
    os.system(f'python {path}')

def main():
    while True:
        print("\n=== Three In Row - Menu ===")
        print("1. Entraîner DQN")
        print("2. Jouer avec DQN")
        print("3. Entraîner Monte Carlo")
        print("4. Jouer avec Monte Carlo")
        print("0. Quitter")

        choice = input("Votre choix : ")

        if choice == "1":
            run_script("dqn_project/train_dqn.py")
        elif choice == "2":
            run_script("dqn_project/play.py")
        elif choice == "3":
            run_script("monte_carlo_project/mc_control.py")
        elif choice == "4":
            run_script("monte_carlo_project/mc_play.py")
        elif choice == "0":
            print("Fermeture du menu.")
            break
        else:
            print("Choix invalide.")

if __name__ == "__main__":
    main()
