# Contributing to LangFair

Welcome and thank you for considering contributing to LangFair!

It takes a lot of time and effort to use software much less build upon it, so we deeply appreciate your desire to help make this project thrive.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Pull Requests](#pull-requests)
2. [Development Setup](#development-setup)
3. [Style Guides](#style-guides)
    - [Code Style](#code-style)
4. [License](#license)

## How to Contribute

### Reporting Bugs

If you find a bug, please report it by opening an issue on GitHub. Include as much detail as possible:
- Steps to reproduce the bug.
- Expected and actual behavior.
- Screenshots if applicable.
- Any other information that might help us understand the problem.

### Suggesting Enhancements

We welcome suggestions for new features or improvements. To suggest an enhancement, please open an issue on GitHub and include:
- A clear description of the suggested enhancement.
- Why you believe this enhancement would be useful.
- Any relevant examples or mockups.

### Pull Requests

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a pull request.

Please ensure your pull request adheres to the following guidelines:
- Follow the project's code style.
- Include tests for any new features or bug fixes.

## Development Setup

1. Clone the repository: `git clone https://github.aetna.com/cvs-health/langfair`
2. Navigate to the project directory: `cd langfair`
3. Create and activate a virtual environment (using `venv` or `conda`)
4. Install dependencies: `poetry install`
5. Install our pre-commit hooks to ensure code style compliance: `pre-commit install`
6. Run tests to ensure everything is working: `pre-commit run --all-files`

You're ready to develop!

**For documentation contributions**
Our documentation lives on the gh-pages branch and is hosted via GitHub Pages.

There are two relevant directories:
* `docs_src` - where source documentation files are located
* `docs` - where the built documentation that is shown on GitHub Pages lives.

To build documentation:
1. Checkout the `gh-pages` branch
2. Navigate to the source dir: `cd docs_src`
3. Build documentation for a GitHub Pages deployment: `make github`

## Style Guides

### Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) to lint and format our files.
- Our pre-commit hook will run Ruff linting and formatting when you commit.
- You can manually run Ruff at any time (see [Ruff usage](https://github.com/astral-sh/ruff#usage)).

Please ensure your code is properly formatted and linted before committing.

## License

Before contributing to this CVS Health sponsored project, you will need to sign the associated [Contributor License Agreement (CLA)](https://forms.office.com/r/gMNfs4yCck)

---

Thanks again for using and supporting LangFair!