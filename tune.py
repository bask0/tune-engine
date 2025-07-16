from engine.cli_interface import CLI


def main():
    cli = CLI()
    cli.tuning_engine.setup(which="all")
    cli.tuning_engine.tune()
    cli.tuning_engine.xval()


if __name__ == "__main__":
    main()
