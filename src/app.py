from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import datetime as dt
from flask import Flask
import usgs_riverdata as usgs

# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
import plotly.graph_objects as go
import math
import numpy as np

server = Flask(__name__)
app = Dash(server=server, external_stylesheets=[dbc.themes.FLATLY])

greatLevel = 700
okLevel = 298
tooLowLevel = 166

cfs_param = "00060"
ft_param = "00065"

reportingParam = cfs_param

if reportingParam == "00060":
    reportinParamLabel = "cfs"
elif reportingParam == "00065":
    reportinParamLabel = "ft"

app.layout = html.Div(
    [
        html.H4("North Fork of the Shenandoah River Levels"),
        dcc.Graph(id="graph"),
        dcc.Interval(
            id="load_interval",
            n_intervals=0,
            max_intervals=0,  # <-- only run once
            interval=1,
        ),
    ]
)


@app.callback(
    Output("graph", "figure"),
    Input(component_id="load_interval", component_property="n_intervals"),
)
def update_chart(slider_range):
    combinedData = getData()

    x_data = pd.to_datetime(combinedData.index, utc=True)
    x_data = x_data.to_pydatetime()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=combinedData["flow_strasburg"],
            mode="lines",
            line=dict(color="blue", width=4),
            name="Strasburg",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=combinedData["flow_mt_jackson"],
            mode="lines",
            line=dict(color="orange", width=4),
            name="Mt. Jackson",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=combinedData["woodstock_flow_est"],
            mode="lines",
            line=dict(color="green", width=4),
            name="Woodstock (estimated)",
        )
    )

    fig.add_hrect(
        y0=greatLevel,
        y1=1500,
        line_width=0,
        fillcolor="green",
        opacity=0.2,
        layer="below",
        annotation_text="Good",
        annotation_position="top right",
    )
    fig.add_hrect(
        y0=tooLowLevel,
        y1=greatLevel,
        line_width=0,
        fillcolor="yellow",
        opacity=0.2,
        layer="below",
        annotation_text="OK",
        annotation_position="top right",
    )
    fig.add_hrect(
        y0=0,
        y1=tooLowLevel,
        line_width=0,
        fillcolor="red",
        opacity=0.2,
        layer="below",
        annotation_text="Too low",
        annotation_position="top right",
    )

    fig.update_yaxes(
        # range=[0, 2000],
        rangemode="nonnegative",
    )

    fig.update_layout(
        # title="Shenandoah County Flow",
        # xaxis_title="X Axis Title",
        yaxis_title="Flow (" + reportinParamLabel + ")",
        # legend_title="Legend Title",
    )

    return fig


def getData():
    STRASBURG_CODE = "01634000"
    MT_JACKSON_CODE = "01633000"

    ndays = 30
    datelist = pd.date_range(end=dt.datetime.today(), periods=ndays).tolist()
    start_date = datelist[0].strftime("%Y-%m-%dT%H:%M")
    end_date = datelist[-1].strftime("%Y-%m-%dT%H:%M")

    strasburg_gage = usgs.gage(
        site_code=STRASBURG_CODE,
        url_params={
            "parameterCd": reportingParam,
            "startDT": start_date,
            "endDT": end_date,
        },
    )
    mt_jackson_gage = usgs.gage(
        site_code=MT_JACKSON_CODE,
        url_params={
            "parameterCd": reportingParam,
            "startDT": start_date,
            "endDT": end_date,
        },
    )

    strasburg_data = strasburg_gage.retrieve(return_pandas=True)
    mt_jackson_data = mt_jackson_gage.retrieve(return_pandas=True)

    flowData = [strasburg_data, mt_jackson_data]

    for dataset in flowData:
        dataset["dateTime"] = pd.to_datetime(dataset["dateTime"])
        dataset.set_index("dateTime", inplace=True)
        dataset.rename(columns={"value": "flow"}, inplace=True)
        dataset["flow"] = pd.to_numeric(dataset["flow"])

    combinedData = strasburg_data.join(
        mt_jackson_data, how="inner", lsuffix="_strasburg", rsuffix="_mt_jackson"
    )

    xcor = np.correlate(
        combinedData["flow_mt_jackson"], combinedData["flow_strasburg"], "same"
    )

    sampleRange = 200
    covOut = [
        combinedData["flow_strasburg"].corr(
            combinedData["flow_mt_jackson"].shift(periods=i)
        )
        for i in range(-sampleRange, sampleRange, 1)
    ]
    maxIdx = np.argmax(covOut)
    sampleShift = maxIdx - sampleRange
    timeShiftHrs = (sampleShift) * 15 / 60

    shiftedData = combinedData.copy()
    shiftedData["flow_mt_jackson"] = shiftedData["flow_mt_jackson"].shift(
        periods=sampleShift
    )

    miles = 60  # taken from https://txpub.usgs.gov/DSS/streamer/web/

    # averageSpeed = miles / timeShiftHrs
    # print(
    #     "The average river speed for the past "
    #     + str(ndays)
    #     + " days is "
    #     + f"{averageSpeed:.2f} mph"
    # )

    flowDiff = combinedData["flow_strasburg"] - combinedData["flow_mt_jackson"].shift(
        periods=sampleShift
    )
    combinedData["flow_diff"] = flowDiff.rolling(window=3).mean()
    flowDiff = flowDiff.dropna()
    averageDiff = flowDiff.mean()

    woodstock_interpolation = 0.75
    tempSeries = (
        combinedData["flow_mt_jackson"].shift(periods=int(sampleShift))
        + combinedData["flow_diff"] * woodstock_interpolation
    )
    combinedData["woodstock_flow_est"] = tempSeries.shift(
        periods=int(sampleShift * (woodstock_interpolation - 1))
    )

    return combinedData


# fig, ax = plt.subplots()

# ax.plot(x_data, combinedData['flow_strasburg'],color='blue',label='Strasburg')
# ax.plot(x_data, combinedData['flow_mt_jackson'],color='orange',label='Mt. Jackson')
# ax.plot(x_data, combinedData['woodstock_flow_est'],color='green',label='Woodstock (estimated)')

# ax.axhline(y=greatLevel, color='g', linestyle='dashed',label='Great floating level')
# ax.axhline(y=okLevel, color='y', linestyle='dashed',label='OK floating level')
# ax.axhline(y=tooLowLevel, color='r', linestyle='dashed',label='Poor floating level')
# ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

# ax.xaxis.set_major_locator(mdates.DayLocator(interval=math.ceil(ndays/20)))
# fig.autofmt_xdate()

# plt.ylabel('Flow ('+reportinParamLabel+')')

# fig.legend(loc='best')
# plt.title('Shenandoah County Flow')
# fig.show()

# fig, ax = plt.subplots()
# ax.plot(combinedData['flow_strasburg']-averageDiff,color='blue',label='Strasburg')
# ax.plot(combinedData['flow_mt_jackson'].shift(periods=sampleShift),color='orange',label='Mt. Jackson')
# # ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

# ax.xaxis.set_major_locator(mdates.DayLocator(interval=math.ceil(ndays/20)))
# fig.autofmt_xdate()

# plt.ylabel('Flow ('+reportinParamLabel+')')

# fig.legend()
# plt.title('Strasburg and Mt. Jackson, Lag = '+str(timeShiftHrs)+' Hrs, offset '+str(averageDiff)+' cfs')
# fig.show()

# shiftedMtJacksonDayLevel = combinedData['flow_mt_jackson'].shift(periods=sampleShift).tail(96).mean()
# strasburgDayLevel = combinedData['flow_strasburg'].tail(96).mean()

# countyInflow=strasburgDayLevel-shiftedMtJacksonDayLevel

# currentLevel=shiftedMtJacksonDayLevel+countyInflow*.75

# print(f'The current estimated Woodstock flow is {currentLevel:.2f} cfs')

# #%%

# lookUpDay=dt.date(2020,6,30)


# if lookUpDay in combinedData.index:
#     dailyAvg=combinedData.resample('D').mean()

#     lookUpData=dailyAvg[dailyAvg.index.date==lookUpDay]

#     lookupInflow=lookUpData['flow_strasburg']-lookUpData['flow_mt_jackson']

#     lookUpLevel=lookUpData['woodstock_flow_est']

#     print ('Flow for '+lookUpDay.strftime("%-m/%-d/%Y")+' was '+str(lookUpLevel[0]))


# %%

if __name__ == "__main__":
    app.run_server(debug=False)
