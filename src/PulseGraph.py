''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
from tokenize import detect_encoding
from bokeh.core.property.dataspec import value
import numpy as np

from math import exp

from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, TextInput, Span
from bokeh.plotting import figure

class LowPassSinglePole:
    def __init__(self, decay):
        self.b = decay
        self.reset()
    def reset(self):
        self.y = 0
    def change_decay(self, decay):
        self.b = decay
    def filter(self, x):
        self.y += (self.b * (x - self.y))
        return self.y

# Set up data
VTHLOW = 0.9
VTHHIGH = 1.6
SAMPLES = 10000
VOLTAGE = 3.3
DETENTS = 10
PULSES_PER_DETENT = 1
RESISTANCE = 10000
CAPACITANCE = 1000
SIGNAL_DURATION = 1 / (DETENTS * PULSES_PER_DETENT) * 2
DELTA_T = SIGNAL_DURATION / SAMPLES

decayDischarging = (DELTA_T / ((RESISTANCE * CAPACITANCE * 1e-9) + DELTA_T)) # Decay between samples (in (0, 1)).
decayCharging = (DELTA_T / ((2 * RESISTANCE * CAPACITANCE * 1e-9) + DELTA_T))
# decay = 0.95

# x = np.linspace(0, 20*np.pi, N)
x = np.linspace(0, SIGNAL_DURATION, SAMPLES)
y = (VOLTAGE / 2) * np.sign(np.sin(2 * np.pi * DETENTS * PULSES_PER_DETENT * x)) + (VOLTAGE / 2)
filter1 = LowPassSinglePole(decayCharging)
vc = []
previous_y = 0
for item in y:
    if item != previous_y:
        if item > 0:
            # charging up, capacitor sees 2R
            filter1.change_decay(decayCharging)
        else:
            # discharging, capacitor sees R
            filter1.change_decay(decayDischarging)
    vc.append(filter1.filter(item))
    previous_y = item

source = ColumnDataSource(data=dict(x=x, y=y))
source2 = ColumnDataSource(data=dict(x=x, y=vc))

# Set up plot
plot = figure(plot_height=400, plot_width=800, title="my square wave",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, SIGNAL_DURATION], y_range=[0, 5.5])

hlineHigh = Span(location=VTHHIGH, dimension='width', line_color='green', line_width=2, line_dash='dashed')
hlineLow = Span(location=VTHLOW * .3, dimension='width', line_color='green', line_width=2, line_dash='dashed')

plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
plot.line('x', 'y', source=source2, line_width=2, line_alpha=0.6)
plot.add_layout(hlineHigh)
plot.add_layout(hlineLow)


# Set up widgets
text = TextInput(title="title", value='my sine wave')
voltageLow = TextInput(title="Low Logic Voltage Threshold (V)", value=str(VTHLOW))
voltageHihg = TextInput(title="High Logic Voltage Threshold (V)", value=str(VTHHIGH))
voltage = Slider(title="Voltage", value=VOLTAGE, start=0, end=5.0, step=0.1)
resistance = Slider(title="Resistance", value=RESISTANCE, start=10, end=20000, step=10)
capacitance = Slider(title="Capacitance", value=CAPACITANCE, start=1, end=10000, step=1)
# freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)
detents = Slider(title="Detents", value = DETENTS, start=1, end=30, step=1)
pulsesPerDetent = Slider(title="Pulses per Detent", value=PULSES_PER_DETENT, start=1, end=2, step=1)
phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)

# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):
    # Get the current slider values
    a = voltage.value
    b = offset.value
    w = phase.value
    d = detents.value
    ppd = pulsesPerDetent.value
    duration = 1 / (d * ppd) * 2
    delta_t = duration / SAMPLES
    thresholdHigh = voltageHihg.value
    thresholdLow = voltageLow.value

    # Generate the new curve
    x = np.linspace(0, duration, SAMPLES)
    y = (a / 2) * np.sign(np.sin(2 * np.pi * d * ppd* x + w)) + a / 2 + b

    decayDischarging = (delta_t / ((resistance.value * capacitance.value * 1e-9) + delta_t)) # Decay between samples (in (0, 1)).
    decayCharging = (delta_t / ((2 * resistance.value * capacitance.value * 1e-9) + delta_t)) # Decay between samples (in (0, 1)).
    filter1 = LowPassSinglePole(decayCharging)

    vc = []
    previous_y = 0
    for item in y:
        if item != previous_y:
            if item > 0:
                # charging up, capacitor sees 2R
                filter1.change_decay(decayCharging)
            else:
                # discharging, capacitor sees R
                filter1.change_decay(decayDischarging)
        vc.append(filter1.filter(item))
        previous_y = item

    source.data = dict(x=x, y=y)
    source2.data = dict(x=x, y=vc)

    # Move the threshold lines
    hlineHigh.set(location=voltageHihg)
    hlineLow.set(location=voltageLow)

    plot.x_range.end = duration

# for w in [offset, amplitude, phase, freq, detents, pulsesPerDetent]:
for w in [voltage, resistance, capacitance, detents, pulsesPerDetent, offset, phase]:
    w.on_change('value', update_data)


# Set up layouts and add to document
# inputs = column(text, offset, amplitude, phase, freq, detents, pulsesPerDetent)
inputs = column(text, voltage, voltageLow, voltageHihg, resistance, capacitance, detents, pulsesPerDetent, offset, phase)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"