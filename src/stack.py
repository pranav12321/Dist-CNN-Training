
import pandas as pd
from bokeh.charts import Bar, output_file, show
x = pd.DataFrame({"gender": ["m","f","m","f","m","f"],
                  "enrolments": [500,20,100,342,54,47],
                  "class": ["comp-sci", "comp-sci",
                            "psych", "psych",
                            "history", "history"]})

bar = Bar(x, values='enrolments', label='class', stack='gender',
         title="Number of students enrolled per class",
         legend='top_right',bar_width=1.0)
output_file("myPlot.html")
show(bar)
