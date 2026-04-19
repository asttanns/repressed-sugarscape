from model import SugarScapeModel
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.components.matplotlib_components import make_mpl_space_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle


# modification: color agents by protest state
# red triangle if protesting, blue circle if not
def agent_portrayal(agent):
    if agent.is_protesting:
        return AgentPortrayalStyle(color="red", marker="^", size=15)
    return AgentPortrayalStyle(color="blue", marker="o", size=8)


# modification: render sugar as yellow and repression as purple overlay
def propertylayer_portrayal(layer):
    if layer.name == "sugar":
        return PropertyLayerStyle(
            color="yellow", alpha=0.7, colorbar=True, vmin=0, vmax=10
        )
    elif layer.name == "repression":
        return PropertyLayerStyle(
            color="purple", alpha=0.5, colorbar=True, vmin=0, vmax=5
        )


sugarscape_space = make_mpl_space_component(
    agent_portrayal=agent_portrayal,
    propertylayer_portrayal=propertylayer_portrayal,
    post_process=None,
    draw_grid=False,
)

GiniPlot = make_plot_component("Gini")
RepressionPlot = make_plot_component("MeanRepression")
ProtestersPlot = make_plot_component("Protesters")
PopulationPlot = make_plot_component("Population")

model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "Random Seed"},
    "width": 50,
    "height": 50,
    "initial_population": Slider(
        "Initial Population", value=200, min=50, max=500, step=10
    ),
    "endowment_min": Slider("Min Initial Endowment", value=25, min=5, max=30, step=1),
    "endowment_max": Slider("Max Initial Endowment", value=50, min=30, max=100, step=1),
    "metabolism_min": Slider("Min Metabolism", value=1, min=1, max=3, step=1),
    "metabolism_max": Slider("Max Metabolism", value=5, min=3, max=8, step=1),
    "vision_min": Slider("Min Vision", value=1, min=1, max=3, step=1),
    "vision_max": Slider("Max Vision", value=5, min=3, max=8, step=1),
    # modification: state-behavior parameters
    "lethality": Slider("Lethality (L)", value=4.0, min=0.5, max=10.0, step=0.5),
    "routineness": Slider("Routineness", value=0.3, min=0.0, max=1.0, step=0.05),
    "repression_increment": Slider(
        "Repression Increment (R)", value=0.5, min=0.0, max=2.0, step=0.1
    ),
    "repression_decay": Slider(
        "Repression Decay (delta)", value=0.05, min=0.01, max=0.5, step=0.01
    ),
    "visibility_threshold": Slider(
        "Visibility Threshold (wealth)", value=30, min=10, max=80, step=5
    ),
}

model = SugarScapeModel()

page = SolaraViz(
    model,
    components=[
        sugarscape_space,
        ProtestersPlot,
        RepressionPlot,
        PopulationPlot,
        GiniPlot,
    ],
    model_params=model_params,
    name="Repressed Sugarscape",
    play_interval=150,
)
page