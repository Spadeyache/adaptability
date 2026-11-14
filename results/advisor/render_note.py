"""Typeset the two-page advisor note from the two packet figures. No model fitting."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

FOLDER = Path(__file__).resolve().parent
INK = "#192d36"
MUTED = "#425964"
TEAL = "#106579"
W, H = 612, 792
LEFT, RIGHT = 54, 558

def line(fig, y, content, size, *, x=LEFT, color=INK, weight="normal", family="DejaVu Sans"):
    fig.text(x/W, 1-y/H, content, fontsize=size, ha="left", va="top",
             family=family, weight=weight, color=color)

def paragraph(fig, y, words, size=10.9, leading=15.9, color=INK):
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family="DejaVu Sans", size=size)
    row = ""
    for word in words.split():
        candidate = f"{row} {word}".strip()
        if row and renderer.get_text_width_height_descent(candidate, font, False)[0] > RIGHT-LEFT-2:
            line(fig, y, row, size, color=color)
            y += leading
            row = word
        else:
            row = candidate
    if row:
        line(fig, y, row, size, color=color)
        y += leading
    return y

def rule(fig, y):
    fig.add_artist(Line2D([LEFT/W, RIGHT/W], [1-y/H,1-y/H], transform=fig.transFigure,
                          color="#a9bec4", linewidth=.75))

def page(number):
    fig = plt.figure(figsize=(W/72, H/72), dpi=72, facecolor="white")
    line(fig, 34, "ROBOT LEARNING  /  RESEARCH NOTE", 9, color=TEAL, weight="bold")
    line(fig, 34, "12 NOVEMBER 2026", 9, x=469, color=MUTED)
    rule(fig, 52)
    rule(fig, 752)
    source = ("forecast/forecast.json" if number == 1
              else "retention/retention.json   ·   results/stage0/retention_check.json")
    line(fig, 763, f"Figure data   ·   results/{source}", 8.5, color=MUTED)
    line(fig, 763, f"{number} / 2", 8.5, x=536, color=MUTED)
    return fig

def figure(fig, stem, y, width):
    img = mpimg.imread(FOLDER.parent / stem / f"{stem}.png")
    height = width * img.shape[0]/img.shape[1]
    ax = fig.add_axes([(W-width)/(2*W), 1-(y+height)/H, width/W, height/H])
    ax.imshow(img)
    ax.set_axis_off()
    return y+height

def heading(fig, y, text):
    line(fig, y, text, 14.5, family="DejaVu Serif", weight="bold")
    return y + 23

def forecast_page():
    fig = page(1)
    line(fig, 71, "What can a robot remember", 26, family="DejaVu Serif")
    line(fig, 104, "about a cube?", 26, family="DejaVu Serif")
    y = paragraph(fig, 151, "A robot can use physical information in two ways. It can use a cube’s properties to predict its motion, and it can remember what earlier attempts revealed about that cube. Two independent simulation studies tested those ideas. Whether either helps the robot estimate a new cube more quickly remains open.")
    y = heading(fig, y+12, "Does supplied physics improve forecasts?")
    y = paragraph(fig, y, "A world model predicts future robot and cube states from the current state and arm commands. We trained one on 36,000 PickCube episodes. An encoder compressed ten known simulator settings, including mass, friction and command delay, into eight input values called context.")
    y = figure(fig, "forecast", y+9, 404)
    y = paragraph(fig, y+6, "Figure 1. The same trained model forecasts the same 4,000 held-out episodes with three context inputs. MSE is the mean squared prediction error after scaling each state variable by its typical training-step change. Error bars are 95% intervals from resampling entire episodes.", 9.2, 13, MUTED)
    y = paragraph(fig, y+10, "Replacing an episode’s context with one from another episode of the same action type raised error by 45.6%. The paired increase was 0.596 (95% interval 0.569–0.624). The training-mean context raised error by 25.3%. The comparison includes free motion and contact, and changing context can change all ten settings.")
    y = paragraph(fig, y+9, "Could prediction error recover the correct settings? In five of six preselected episodes, the best prediction occurred at the supplied context. In the sixth, a scripted push, moving 10% toward the other episode’s context slightly improved the forecast. The study missed its six-of-six criterion and left inference from motion unverified.")
    assert y < 742, y
    return fig

def retention_page():
    fig = page(2)
    line(fig, 71, "Remembering one cube", 26, family="DejaVu Serif")
    y = paragraph(fig, 118, "A separate experiment used a mass estimator trained on labeled lift attempts. It read joint motion, finger compression and arm commands. Every 25 steps, a filter combined the latest estimate with earlier evidence. The arm used a fixed 40 g guess in every attempt. We included all 60 attempts.")
    y = paragraph(fig, y+7, "We replayed identical observations through two copies of the filter. One restarted from its original guess at every attempt. The other carried its mass estimate and uncertainty forward. Both started with the same guess on trial one.")
    y = figure(fig, "retention", y+8, 373)
    y = paragraph(fig, y+5, "Figure 2. Each of 12 independent sequences contained five fresh starts with a 400 g cube at friction 0.4. Lines show average end-of-attempt error in grams. Bands are 95% intervals from resampling whole five-attempt sequences, keeping the two filters paired.", 9.2, 13, MUTED)
    y = paragraph(fig, y+9, "At trial five, mean absolute error was 37.4 g with memory and 83.5 g after resetting. The paired advantage was 46.2 g (95% interval 14.6–73.7). Error with memory rose slightly from trial two to three. Its change from trial one to five was −16.1 g (−41.2 to +5.7), so this study leaves the size of that improvement uncertain.")
    y = paragraph(fig, y+6, "A retrospective check used the original ±40 g threshold. On trials two to five, 28/48 carried and 13/48 reset estimates stayed within range through the end for at least ten steps.")
    y = heading(fig, y+11, "Can experience improve the first guess on a new cube?")
    y = paragraph(fig, y, "First establish that a frozen world model can recover mass and friction from pushes and lifts on unseen cubes. Then introduce fresh cubes one at a time from a population enriched in heavy cubes. Compare a fixed starting guess with one updated only from previously encountered cubes. Give both the same interactions, vary cube order and use independent test sequences. Set the accuracy threshold in advance. Report first-guess error, steps to threshold and the fraction that never reach it.")
    assert y < 742, y
    return fig

if __name__ == "__main__":
    plt.rcParams.update({"pdf.fonttype": 42, "figure.dpi": 72})
    with PdfPages(FOLDER/"research_note.pdf", metadata={"Title": "What can a robot remember about a cube?", "Creator": "Matplotlib"}) as document:
        for make_page in (forecast_page, retention_page):
            page_figure = make_page()
            document.savefig(page_figure, dpi=240)
            plt.close(page_figure)
