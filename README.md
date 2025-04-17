# Der Roboterlehrer: Interpretable and deterministic MQM-inspired translation evaluation

[![CLicense](https://img.shields.io/badge/License%20-%20MIT%20-%20%23ff6863?style=flat)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE) [![Python 3.10](https://img.shields.io/badge/Python%20-%203.10%20-%20?style=flat&logo=python&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Issues](https://img.shields.io/github/issues/aphil311/tiny-bs?style=flat&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)


<!-- ABOUT THE PROJECT -->
## About The Project
We seek to create a chatbot capable of performing multidimensional translation evaluation with feedback without making any LLM API calls. We hope that this approach is more interpretable and deterministic than existing state-of-the-art.

At the moment we only support the German-English language pair.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started


### Installation
1. Clone this repository with `git clone https://github.com/aphil311/teach-bs.git`.
2. Install the dependencies with `pip install -r requirements.txt`.
   - You must downgrade to `pip < 24.1` with `pip install pip=24.0` to install `laser_encoders`.
   - You can upgrade after installing.


### Usage 
1. Run the streamlit app with `streamlit run app.py`.
2. The chatbot will immediately prompt you with a German to English translation.
    - You can switch to English to German on the sidebar.
3. Scores are computed as the raw arithmetic mean and can be found in the 'scores' sidebar tab.


<p align="right">(<a href="#readme-top">back to top</a>)</p> 



<!-- ROADMAP -->
## Roadmap

<!-- - [X] **Build the model**
- [ ] **Training**
- [ ] **Validation**
- [ ] **Evaluation** -->


See the [open issues](https://github.com/aphil311/talos/issues) for a full list of proposed features (and known issues).



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
I would like to thank Professor Srinivas Bangalore as well as the TRA 301 TAs their for their invaluable guidance, feedback, and support.

<p align="right">(<a href="#readme-top">back to top</a>)</p>