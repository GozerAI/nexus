Thank you for your interest in contributing!
This project is dual-licensed (AGPL-3.0 for open-source use + proprietary for commercial use) and uses a Contributor License Agreement (CLA) to ensure that contributions can be incorporated under both licenses.

This document explains the expectations, requirements, and workflow for all contributors—individual or corporate.

1. Before You Begin
  Please familiarize yourself with the following documents located under the legal/ directory:
    Licensing
      - Open Source License: legal/licenses/AGPL-3.0.txt
      - Commercial License: legal/licenses/COMMERCIAL-LICENSE.txt
    Contributor Agreements (Required)
      - Commercial Contributor Agreement (CCA): legal/agreements/CCA.md
      - Contributor Acknowledgement Form: legal/agreements/CONTRIBUTOR-ACKNOWLEDGEMENT.md
      - Employer Contributor Agreement (if applicable): legal/agreements/EMPLOYER-CCA.md

  All contributors must read and agree to the Contributor Acknowledgement and the CCA before submitting a pull request.

  If you are contributing as part of your job or using your employer’s equipment/resources, your employer must also sign the Employer CCA.

  Signature is handled automatically during the GitHub Pull Request process via the CLA Bot (see below).

2. Contributor License Agreement (CLA)

This repository uses CLA Assistant to track contributor signatures.
When you submit your first pull request, the CLA bot will:
  - Check whether you have signed the CCA
  - If not, it will comment on your PR with instructions

You sign by replying:

      I have read the CLA Document and I hereby sign the CLA

After signing, the CLA bot updates the PR status

  Your pull request cannot be merged until all required signatures are complete.

The CLA ensures:
  - Your contribution can be used under both AGPL-3.0 and the commercial license
  - You retain ownership of your contributions
  - You grant the project a license to use your contributions consistently

3. Expectations for All Contributors

  Read and understand the legal requirements
  You must confirm that:
  - You have read the CCA
  - You have read the Contributor Acknowledgement
  - You understand the dual-licensing structure
  - If you are contributing on behalf of an employer, they have executed the Employer CCA

  These confirmations are included as mandatory checkboxes in the PR template: .github/pull_request_template.md

4. Code Quality & Style Requirements

  To maintain a consistent codebase, contributions must follow these standards:

    4.1 Python Style
      - Use type hints for all public functions and methods
      - Follow PEP8 conventions
      - Code must pass the project’s linters before submission (configured in pyproject.toml)

    4.2 Testing
      - All contributions must include unit tests under tests/

      New functionality must include both:
      - Passing tests for happy paths
      - Failing tests for invalid or error cases

    4.3 Documentation
      - Public functions/classes must include clear docstrings

      Modifying behavior? Update relevant documentation in: 
      - README.md
      - Any module-level documentation
      - Example usage files (if applicable)

    4.4 No breaking changes without discussion
      Open an issue to propose:
      - API changes
      - Backwards-incompatible behavior
      - Feature removals

5. Workflow for Contributing

    Step 1: Fork the repository
      git clone https://github.com/'<your-username>'/'<repo>'.git

    Step 2: Create a feature branch
      git checkout -b feature/my-new-feature
  
    Step 3: Make your changes
      Follow the coding, testing, and documentation guidelines above.

    Step 4: Run tests and linters locally
      pytest
      ruff check .

    Step 5: Submit a Pull Request
      Your pull request must include:
        - A clear description of the change
        - Filled-out legal & licensing checkboxes
        - Associated tests
        - Confirmation via the CLA bot

      If contributing on behalf of an employer, the PR must show:
        - Employer CCA signed
        - You are authorized to contribute employer-owned IP

7. Licensing of Contributions

  By contributing to this project:
    - You retain ownership of your contribution
    - You grant the project a dual license right under:
        - AGPL-3.0 (open source)
        - Commercial proprietary license (closed source)

  This is required in order to:
    - Distribute binaries
    - Bundle your contributions into paid offerings
    - Maintain consistent licensing across the codebase
    - Avoid fragmented or incompatible rights

  This ensures your work can be part of both the open and commercial editions.

  Full details are in:
  legal/agreements/CCA.md

  7. Security, Ethical, and Behavioral Expectations

  All contributors are expected to:
    - Avoid introducing malicious, harmful, or deceptive code
    - Follow standard security best practices
    - Clearly disclose any vulnerabilities discovered
    - Behave respectfully and collaboratively within issues and PRs
    - No harassment, abuse, or discriminatory behavior will be tolerated.

8. Where to Get Help or Ask Questions

  If you are unsure about:
    - The contribution process
    - Licensing requirements
    - Employer obligations
    - Technical design decisions
    - Roadmap or priorities

Please open an issue on GitHub or start a discussion in the repo’s Discussions section.
OR
Email the maintainer at chris@gozerai.com

