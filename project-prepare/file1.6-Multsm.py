# Import required libraries
from openai import OpenAI
import os
import httpx
import json
import re
import time

# ====== Helper Functions ======
# Remove numbering and other prefixes
def clean_line_prefix(line):
    return re.sub(r'^\s*(\d+\.\s*|\d+[\.\u3001Z\s]*|[\u2460-\u2469]|[一二三四五六七八九十][、\.]?)\s*', '', line).strip()

# Remove English, numbers, symbols, keep only Chinese characters
def remove_noise(line):
    return ''.join(re.findall(r'[\u4e00-\u9fa5]+', line))

# ====== Example Question List (only one test retained, can be expanded) ======
questions = [
["How does physics affect social development? Please analyze by combining mathematical models and considering biological factors?", ["Biology", "Sociology", "Physics", "Mathematics"]],
["How does the application of geographic information systems in agricultural planning reflect the combination of mathematical statistics and chemical analysis?", ["Geography", "Agriculture", "Mathematics", "Chemistry"]],
["Taking the COVID-19 pandemic as an example, how do medical resource allocation models integrate sociological equity principles and mathematical optimization algorithms?", ["Medical", "Sociology", "Mathematics"]],
["Mathematical modeling methods and historical evolution of quantum mechanics principles in biological photosynthesis research", ["Physics", "Biology", "Mathematics", "History"]],
["What is the analogous relationship between chaotic phenomena in financial markets and nonlinear systems in physics? Please explain with mathematical models", ["Finance", "Physics", "Mathematics"]],
["What geographical, chemical, and mathematical methods need to be integrated in simulating the impact of the Qinghai-Tibet Plateau uplift on climate?", ["Geography", "Chemistry", "Mathematics"]],
["How do antibiotic resistance transmission models combine sociological behavior analysis and biological network computing?", ["Biology", "Medical", "Sociology", "Mathematics"]],
["From historical documents, how did ancient Chinese water conservancy projects reflect the combination of physical mechanics and agricultural practices?", ["History", "Physics", "Agriculture"]],
["Mathematical integration methods of geographic remote sensing data and chemical pollutant diffusion models in urban heat island effect assessment", ["Geography", "Chemistry", "Mathematics"]],
["How does group dynamics in sociology draw on physics phase transition theory and mathematical differential equations?", ["Sociology", "Physics", "Mathematics"]],
["How does mathematical modeling of photosynthetic quantum efficiency promote the design of agricultural photobioreactors?", ["Biology", "Agriculture", "Mathematics", "Physics"]],
["What are the similarities and differences between Monte Carlo simulation in financial risk assessment and physical particle transport models?", ["Finance", "Physics", "Mathematics"]],
["What chemical and mathematical methods need to be integrated into agricultural pest early warning models based on geographic information systems?", ["Geography", "Agriculture", "Chemistry", "Mathematics"]],
["How do bioinformatics models for virus mutation prediction combine sociological transmission network analysis?", ["Biology", "Medical", "Sociology", "Mathematics"]],
["From a historical perspective, how did alchemy promote the early integration of chemistry and physical metallurgy technology?", ["History", "Chemistry", "Physics"]],
["What biological anatomy and physical imaging principles need to be integrated into deep learning models for medical image segmentation?", ["Medical", "Biology", "Physics", "Mathematics"]],
["What mathematical relationship exists between financial derivative pricing models and diffusion equations in physics?", ["Finance", "Physics", "Mathematics"]],
["How do soil heavy metal pollution remediation schemes integrate geographic spatial analysis and chemical stabilization technologies?", ["Geography", "Agriculture", "Chemistry"]],
["How does the phase transition model of social opinion dissemination draw on the Ising model in physics?", ["Sociology", "Physics", "Mathematics"]],
["What biological, climatic, and geographical factors need to be considered in mathematical models for predicting agricultural pest populations?", ["Agriculture", "Biology", "Geography", "Mathematics"]],
["How does the application of nuclear magnetic resonance technology in biomedicine reflect the combination of physical principles and chemical analysis?", ["Physics", "Chemistry", "Medical"]],
["What social network analysis and mathematical statistical methods are needed for financial systemic risk monitoring?", ["Finance", "Sociology", "Mathematics"]],
["How do ice core data in paleoclimate reconstruction reflect the correlation between chemical composition and geographical latitude?", ["Geography", "Chemistry", "History"]],
["How do multi-scale models for protein folding simulation integrate physical molecular dynamics and mathematical optimization algorithms?", ["Biology", "Physics", "Mathematics"]],
["What chemical fertilizer dynamics and geographical environment parameters need to be integrated into crop growth models in smart agriculture?", ["Agriculture", "Chemistry", "Geography", "Mathematics"]],
["How do optimization models for spatio-temporal allocation of medical resources balance social equity and logistics efficiency?", ["Medical", "Sociology", "Geography", "Mathematics"]],
["From historical document analysis, how did Silk Road trade promote agricultural species migration and geographical cognitive development?", ["History", "Agriculture", "Geography"]],
["How do high-frequency trading strategies in financial markets apply stochastic process theory from physics?", ["Finance", "Physics", "Mathematics"]],
["What chemical fingerprint identification and geographical diffusion models are needed for source analysis of atmospheric PM2.5 pollution?", ["Chemistry", "Geography", "Mathematics"]],
["How does research on urban spatial differentiation in sociology combine geographic spatial analysis and statistical modeling?", ["Sociology", "Geography", "Mathematics"]],
["Mathematical modeling methods and experimental verification of quantum biology in enzyme-catalyzed reactions", ["Physics", "Biology", "Mathematics", "Chemistry"]],
["How do variable rate fertilization systems in precision agriculture integrate geographic GIS data and crop nutrition mathematical models?", ["Agriculture", "Geography", "Mathematics", "Chemistry"]],
["What sociological behavioral factors and chemical degradation mechanisms need to be considered in life cycle assessment of medical waste treatment?", ["Medical", "Sociology", "Chemistry"]],
["How did the development of ancient navigation technology promote the interaction between geographical discoveries and historical processes?", ["History", "Geography", "Physics"]],
["How do financial insurance actuarial models draw on life table analysis methods in biology?", ["Finance", "Biology", "Mathematics"]],
["How to quantify the role of microorganisms in soil carbon cycle models through mathematical network analysis?", ["Agriculture", "Biology", "Mathematics", "Chemistry"]],
["What geographical accessibility and social equity indicators need to be balanced in hospital location optimization models?", ["Medical", "Geography", "Sociology", "Mathematics"]],
["Analysis of material cycle efficiency in agricultural ecosystems from the perspective of physics energy conservation", ["Agriculture", "Physics", "Biology", "Mathematics"]],
["How does molecular docking simulation in chemical drug research and development combine biological receptor dynamics and mathematical optimization?", ["Chemistry", "Biology", "Medical", "Mathematics"]],
["How do mathematical models for social media public opinion monitoring reflect the characteristics of sociological group behavior?", ["Sociology", "Mathematics", "Computer"]],
["Geographical channel analysis of historical plague transmission and evolutionary correlation with modern infectious disease models", ["History", "Geography", "Medical", "Mathematics"]],
["What physical fluctuation theories and mathematical methods are needed for volatility surface fitting in financial derivatives markets?", ["Finance", "Physics", "Mathematics"]],
["How does molecular design for crop genetic improvement integrate biological gene networks and agricultural phenotypic mathematical models?", ["Agriculture", "Biology", "Mathematics"]],
["What physical dose distribution and biological tissue responses need to be considered in medical radiotherapy planning systems?", ["Medical", "Physics", "Biology"]],
["How does intergenerational mobility research in sociology apply mathematical Markov chain models?", ["Sociology", "Mathematics"]],
["Chemical pollution factor correction methods for geographically weighted regression in urban housing price analysis", ["Geography", "Chemistry", "Mathematics", "Finance"]],
["Physical implementation and mathematical algorithm challenges of quantum computing in drug molecular simulation", ["Physics", "Chemistry", "Biology", "Mathematics"]],
["What chemical water-saving technologies need to be balanced with geographical spatial distribution in agricultural water resource management?", ["Agriculture", "Geography", "Chemistry"]],
["How do hospital infection control models combine sociological interpersonal contact networks and biological transmission dynamics?", ["Medical", "Sociology", "Biology", "Mathematics"]],
["From historical archives, how did physics breakthroughs during the Industrial Revolution promote the industrialization of chemistry?", ["History", "Physics", "Chemistry"]],
["How does sociological feature engineering in financial credit scoring models optimize mathematical classification algorithms?", ["Finance", "Sociology", "Mathematics"]],
["What chemical kinetics and geographical environment parameters need to be integrated into atmospheric ozone layer hole repair plans?", ["Chemistry", "Geography", "Physics"]],
["How does mathematical modeling of biological clock rhythms reveal physical temperature compensation mechanisms and chemical regulatory networks?", ["Biology", "Physics", "Chemistry", "Mathematics"]],
["How do wearable device data in smart healthcare combine biological signal processing and social behavior analysis?", ["Medical", "Biology", "Sociology", "Mathematics"]],
["What sociological farmer decision-making behaviors need to be considered in game models for agricultural pest resistance management?", ["Agriculture", "Sociology", "Biology", "Mathematics"]],
["The impact of geographical boundary division on historical and cultural identity: balancing mathematical zoning algorithms and sociological indicators", ["Geography", "History", "Sociology", "Mathematics"]],
["How do graph neural networks in financial anti-fraud systems integrate sociological relationship chains and mathematical representation learning?", ["Finance", "Sociology", "Mathematics"]],
["How do temperature sensitivity models of soil organic matter decomposition integrate biological enzyme kinetics and chemical stabilization mechanisms?", ["Agriculture", "Biology", "Chemistry", "Mathematics"]],
["What physics queuing theory and geographical spatial layout need to be combined in hospital emergency department process optimization?", ["Medical", "Physics", "Geography", "Mathematics"]],
["Physical limits and mathematical reconstruction methods of quantum sensing technology in biomedical imaging", ["Physics", "Biology", "Mathematics", "Medical"]],
["How to realize agricultural climate risk zoning through coupling of geographical climate data and crop mathematical models?", ["Agriculture", "Geography", "Mathematics"]],
["How to quantify sociological patient satisfaction into mathematical indicators in medical quality evaluation systems?", ["Medical", "Sociology", "Mathematics"]],
["From historical document analysis, how did China's four great inventions in ancient times promote the development of physics and chemistry disciplines?", ["History", "Physics", "Chemistry"]],
["What physical phase transition theories and mathematical catastrophe models are needed for extreme risk early warning in financial markets?", ["Finance", "Physics", "Mathematics"]],
["How does mathematical modeling of biological cell communication reveal the synergy between chemical signal transduction and physical force conduction?", ["Biology", "Chemistry", "Physics", "Mathematics"]],
["How does land use classification in geographical national conditions surveys combine chemical remote sensing spectroscopy and mathematical clustering algorithms?", ["Geography", "Chemistry", "Mathematics"]],
["What biological soil processes and mathematical accounting methods need to be integrated into the assessment of carbon sequestration effects of agricultural conservation tillage?", ["Agriculture", "Biology", "Mathematics"]],
["How do hospital building energy-saving designs balance physics energy consumption simulation and geographical microclimate conditions?", ["Medical", "Physics", "Geography", "Mathematics"]],
["What geographic spatial analysis and mathematical statistical tools are needed for research on urban community differentiation in sociology?", ["Sociology", "Geography", "Mathematics"]],
["Chemical reaction path optimization and mathematical modeling of quantum biology in photosynthesis research", ["Physics", "Chemistry", "Biology", "Mathematics"]],
["How does multi-omics data integration in precision medicine combine biological network analysis and mathematical dimensionality reduction techniques?", ["Medical", "Biology", "Mathematics"]],
["How does blockchain application in financial insurance technology reflect the combination of sociological trust mechanisms and mathematical cryptography?", ["Finance", "Sociology", "Mathematics"]],
["What chemical environmental factor corrections are needed for the application of geospatial big data in epidemic prevention and control?", ["Geography", "Chemistry", "Medical", "Mathematics"]],
["How do crop growth prediction models in agricultural Internet of Things systems integrate biological sensors and mathematical regression algorithms?", ["Agriculture", "Biology", "Mathematics", "Computer"]],
["What physics imaging principles and biomedical knowledge need to be combined in AI systems for medical image diagnosis?", ["Medical", "Physics", "Biology", "Mathematics"]],
["How does research on educational opportunity equity in sociology apply mathematical causal inference methods?", ["Sociology", "Mathematics", "Education"]],
["Integration of geographical trajectory analysis and chemical source tracing models for long-distance transport of atmospheric pollutants", ["Geography", "Chemistry", "Mathematics"]],
["How does mathematical modeling of biological metabolic networks guide chemical production optimization in synthetic biology?", ["Biology", "Chemistry", "Mathematics", "Engineering"]],
["Sociological compliance behavior analysis and mathematical risk measurement models in financial regulatory technology", ["Finance", "Sociology", "Mathematics"]],
["From historical climate data, how did geographical environment changes affect the rise and fall of agricultural civilizations?", ["History", "Geography", "Agriculture"]],
["How does the application of quantum computing in cryptography change financial security systems and mathematical algorithm foundations?", ["Finance", "Physics", "Mathematics", "Computer"]],
["What geographical accessibility and sociological indicators are needed for evaluating the fairness of regional allocation of medical resources?", ["Medical", "Geography", "Sociology", "Mathematics"]],
["How to optimize chemical conversion paths of agricultural straw comprehensive utilization through biological catalysts?", ["Agriculture", "Chemistry", "Biology"]],
["Spatial isolation design for hospital infection control needs to combine physics aerodynamics and geographical spatial analysis", ["Medical", "Physics", "Geography", "Mathematics"]],
["How does organizational network analysis in sociology apply mathematical graph theory and computer simulation?", ["Sociology", "Mathematics", "Computer"]],
["Sociological demographic factor correction methods for geographically weighted regression in real estate valuation", ["Geography", "Sociology", "Mathematics", "Finance"]],
["Coupling of mathematical modeling of biofilm formation processes with physical fluid dynamics and chemical signal transduction", ["Biology", "Physics", "Chemistry", "Mathematics"]],
["How do UAV remote sensing data in precision agriculture combine chemical vegetation indices and mathematical interpolation algorithms?", ["Agriculture", "Chemistry", "Geography", "Mathematics"]],
["Balancing sociological considerations and mathematical interpretability modeling in medical artificial intelligence ethics", ["Medical", "Sociology", "Mathematics"]],
["From historical archives, how did the Silk Road promote geographical cognition and agricultural species exchange?", ["History", "Geography", "Agriculture"]],
["How does sociological group behavior analysis in financial markets apply physics self-organized criticality theory?", ["Finance", "Sociology", "Physics", "Mathematics"]],
["What biological accumulation models and chemical speciation analyses are needed for evaluating phytoremediation efficiency of soil heavy metal pollution?", ["Agriculture", "Biology", "Chemistry", "Mathematics"]],
["Multi-objective optimization of physical simulation and geographical climate adaptive design for hospital building energy conservation", ["Medical", "Physics", "Geography", "Mathematics"]],
["How do cultural transmission models in sociology draw on physics diffusion equations and mathematical partial differential methods?", ["Sociology", "Physics", "Mathematics"]],
["What sociological poverty indicators and mathematical methods are needed for the application of geospatial analysis in targeted poverty alleviation?", ["Geography", "Sociology", "Mathematics"]],
["Mathematical modeling of circadian gene expression reveals the synergistic mechanism of physical temperature compensation and chemical modification", ["Biology", "Physics", "Chemistry", "Mathematics"]],
["Mathematical integration of chemical process models of agricultural greenhouse gas emissions and biological soil respiration", ["Agriculture", "Chemistry", "Biology", "Mathematics"]],
["How does Six Sigma management for medical quality improvement combine sociological patient experience and mathematical process optimization?", ["Medical", "Sociology", "Mathematics"]],
["Synergy between physical limits of quantum sensing in biomedical detection and chemical labeling technology", ["Physics", "Chemistry", "Biology", "Medical"]],
["How do financial anti-money laundering monitoring systems integrate sociological transaction networks and mathematical anomaly detection algorithms?", ["Finance", "Sociology", "Mathematics"]],
["Combination of chemical pollution hotspot identification and mathematical statistical methods in geographical national conditions monitoring", ["Geography", "Chemistry", "Mathematics"]],
["What geographical path planning and mathematical linear programming methods are needed for hospital logistics material distribution optimization?", ["Medical", "Geography", "Mathematics"]],
["How does research on urban spatial justice in sociology apply mathematical spatial econometric methods?", ["Sociology", "Geography", "Mathematics"]],
["Chemical reaction path optimization and mathematical constraint modeling in biological metabolic engineering", ["Biology", "Chemistry", "Mathematics", "Engineering"]],
["Geospatial accounting and mathematical statistical methods for evaluating agricultural ecosystem service values", ["Agriculture", "Geography", "Mathematics"]],
["Reliability verification of medical AI-assisted diagnosis needs to combine physical measurement uncertainty and mathematical confidence interval analysis", ["Medical", "Physics", "Mathematics"]],
["From historical documents, how did ancient Chinese astronomical observations affect geographical navigation and agricultural calendars?", ["History", "Geography", "Agriculture"]],
["Evolution of physical analogy models and mathematical stochastic differential equations in financial derivatives markets", ["Finance", "Physics", "Mathematics"]],
["Coupling of mathematical modeling of biological cell autophagy processes with chemical regulation and physical force conduction", ["Biology", "Chemistry", "Physics", "Mathematics"]],
["Integrated analysis of chemical pollution exposure and sociological risk perception in geographical vulnerability assessment", ["Geography", "Chemistry", "Sociology", "Mathematics"]],
["How do crop phenomics data in smart agriculture combine biological genetic models and mathematical correlation analysis?", ["Agriculture", "Biology", "Mathematics"]],
["Multi-agent modeling of physical evacuation simulation and sociological behavior decision-making in hospital emergency management", ["Medical", "Physics", "Sociology", "Mathematics"]],
["How does intergenerational mobility research in sociology apply mathematical random forest algorithms and social network analysis?", ["Sociology", "Mathematics"]],
["Mathematical integration of chemical fingerprint identification and geospatial inversion for atmospheric pollutant source apportionment", ["Chemistry", "Geography", "Mathematics"]],
["Verification of enzymatic reaction tunneling effect and mathematical probability models in quantum biology", ["Physics", "Biology", "Chemistry", "Mathematics"]],
["Synergy between sociological medical insurance payment reform and mathematical actuarial models for medical cost control", ["Medical", "Sociology", "Mathematics"]],
["Chemical life cycle assessment and geospatial mathematical models for agricultural carbon footprint accounting", ["Agriculture", "Chemistry", "Geography", "Mathematics"]],
["Integration and optimization of physical fluid simulation of hospital building ventilation systems with geographical microclimate data", ["Medical", "Physics", "Geography", "Mathematics"]]]

# ====== Model Configuration ======
models_config = {
    "gemini-2.5-flash-preview-04-17-thinking": {
        "api_key": "sk-VJrRRrYljSfcLQPKD2ocOw8NrKaFOPsTszZy1gb5qWJixq2Y",
        "base_url": "https://api.aigptapi.com/v1/"
    },
    "grok-3-beta": {
        "api_key": "sk-VJrRRrYljSfcLQPKD2ocOw8NrKaFOPsTszZy1gb5qWJixq2Y",
        "base_url": "https://api.aigptapi.com/v1/"
    },
    "doubao-pro-256k": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "moonshot-v1-8k": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "deepseek-v3": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "hunyuan-turbo": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    },
    "qwen2.5-72b-instruct": {
        "api_key": "sk-N4rH9BjW8xR1akf0C01426F958D74c9d97Bd7a131a09B5B4",
        "base_url": "https://api.vansai.cn/v1"
    }
}

# ====== Main Processing Flow ======
all_model_results = []

for model_name, config in models_config.items():
    print(f"\n🚀 Starting to use model: {model_name}")
    model_results = []

    api_client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        http_client=httpx.Client(verify=False)
    )

    for idx, core_question in enumerate(questions):
        print(f"\n{'='*40}\n  🔍 Question {idx+1}: {core_question[0]}\n{'='*40}")

        try:
            domains_list = core_question[1]
            sum_list = []

            for domain in domains_list:
                print(f"\n🌐 Processing domain: {domain}")

                def request_func():
                    return api_client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": rf"""From the perspective of {domain}, what are the most important key factors surrounding " \"{core_question[0]}\" "?\nOutput only the factors themselves, without any preceding analytical statements, one per line, deduplicated and sorted by importance. Output strictly in the following format: Factor A Factor B"""
                        }]
                    )

                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        response = request_func()
                        tasks3 = response.choices[0].message.content

                        # Cleaning: remove prefixes + remove English/symbols + remove empty
                        topic_list = [domain] + [
                            factor for factor in [
                                remove_noise(clean_line_prefix(line))
                                for line in tasks3.split("\n") if line.strip()
                            ] if factor
                        ]

                        sum_list.append(topic_list)

                        print(f"    ✅ Extracted {len(topic_list)-1} key factors:")
                        for factor in topic_list[1:]:
                            print(f"      - {factor}")

                        break
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            print(f"  ⚠️ Retrying ({attempt+1}/{max_attempts}): {e}")
                            time.sleep(1)
                        else:
                            raise e

            model_results.append({
                "question_index": idx + 1,
                "core_question": core_question,
                "tasks3": tasks3,
                "domains_list": domains_list,
                "sum_list": sum_list
            })

            print(f"\n✅ Completed question {idx+1}, extracted a total of {sum(len(x)-1 for x in sum_list)} factors")

        except Exception as e:
            print(f"  ❌ Processing failed: {e}")
            model_results.append({
                "question_index": idx + 1,
                "core_question": core_question,
                "error": str(e)
            })

    all_model_results.append({
        "model_name": model_name,
        "results": model_results
    })

    print(f"\n{'*'*40}\n✅ All questions processed for model {model_name}\n{'*'*40}")

# ====== Save Results ======
output_dir = r"D:\\project"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "generated_results_multi_model.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_model_results, f, ensure_ascii=False, indent=4)

print(f"\n🎉 All model questions processed, results saved to: {output_file}")
