from project import CLI


def main():
    """
    Initializes the CLI, sets up the tuning engine for all configurations, performs tuning, and executes cross-validation.
    This function orchestrates the following steps:
    1. Instantiates the CLI interface.
    2. Sets up the tuning engine with all available options.
    3. Runs the tuning process.
    4. Performs cross-validation on the tuned models.
    """
    
    cli = CLI()
    cli.tuning_engine.setup(which="all")
    cli.tuning_engine.tune()
    cli.tuning_engine.xval()


if __name__ == "__main__":
    main()
