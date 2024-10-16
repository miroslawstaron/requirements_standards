import os
import filter_docs

# Global variables
menu_open = True
folder_path = "standards/22_standards"
output_csv = "outputs/latency_paragraphs.csv"
keywords = ["latency", "latencies"]
ignored_sections = ["References", "Appendix", "Definitions", "Abbreviations"]
model_name = "llama3.1:70b"
verbose = False

def display_menu():
    print("Please select an option:")
    print("1. Option 1: Configure the pipeline")
    print("2. Option 2: Execute the pipeline")
    print("3. Option 3: Help")
    print("4. Exit")

def display_config_menu():
    print("Please select an option:")
    print("1. Set folder path")
    print("2. Set output CSV")
    print("3. Set keywords")
    print("4. Set ignored sections")
    print("5. Set model name")
    print("6. Set verbose")
    print("7. Print configuration")
    print("8. Exit")

def configure_pipeline():
    global folder_path, output_csv, keywords, ignored_sections, model_name, verbose
    while True:
        display_config_menu()
        choice = input("Enter your choice (1-8): ")
        os.system('cls' if os.name == 'nt' else 'clear')
        if choice == '1':
            folder_path = input("Enter the folder path: ")
        elif choice == '2':
            output_csv = input("Enter the output CSV: ")
        elif choice == '3':
            keywords = input("Enter the keywords (comma-separated): ").split(",")
        elif choice == '4':
            ignored_sections = input("Enter the ignored sections (comma-separated): ").split(",")
        elif choice == '5':
            model_name = input("Enter the model name: ")
        elif choice == '6':
            verbose_input = input("Enter verbose (True/False): ").lower()
            verbose = verbose_input == 'true'
        elif choice == '7':
            print_config()
        elif choice == '8':
            print("Exiting the configuration menu.")
            break
        else:
            print("Invalid choice. Please select a valid option (1-8).")

def print_config():
    print("Current Configuration:")
    print(f"Folder path: {folder_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Keywords: {keywords}")
    print(f"Ignored sections: {ignored_sections}")
    print(f"Model name: {model_name}")
    print(f"Verbose: {verbose}")

def main():
    global menu_open
    while menu_open:
        display_menu()
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            configure_pipeline()
        elif choice == '2':
            print("Executing Pipeline...")
            print("Executing filtering...")

            try:
                filter_docs.execute_filtering(folder_path, keywords, output_csv, ignored_sections, model_name, verbose)
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Pipeline failed.")
                continue

            print("Executing requirement Generating...")
            print("ERROR: (Placeholder because requirement generating is not implemented yet for this branch)")

            try:
                pass
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Pipeline failed.")
                continue

            # Placeholder for requirement generating
            print("Pipeline executed successfully.")

        elif choice == '3':
            print("Help menu (placeholder).")
            # Placeholder for help menu
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            menu_open = False
        else:
            print("Invalid choice. Please select a valid option (1-4).")

if __name__ == "__main__":
    main()
