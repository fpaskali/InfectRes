library(shiny)
library(shinythemes)
library(shinyjs)
library(DT)
library(ggplot2)
library(mgcv)

cal_ui <- function(request) {
  tagList(
    fluidPage(
      theme = shinytheme("sandstone"),
      useShinyjs(),
      titlePanel("InfectResonator Calibration"),
      tabsetPanel(id = "tabs",
                  ## Start of Tab Data
                  tabPanel("Features Data", value = "tab3",
                           sidebarLayout(
                             sidebarPanel(
                               h5("You can also upload existing features data and go to experiment info", style="font-weight:bold"),
                               fileInput("intensFile", "Select CSV file",
                                         multiple = FALSE,
                                         accept = c("text/csv",
                                                    "text/comma-separated-values,text/plain",
                                                    ".csv")), 
                               hr(style="border-color: gray"),
                               actionButton("expInfo", label = "Switch To Experiment Info"),
                               hr(style="border-color: gray"),
                               h5("For restart with new data", style="font-weight:bold"),
                               actionButton("deleteData", label = "Delete Data"), br(),
                             ),
                             mainPanel(
                               DTOutput("intens")
                             )
                           )
                  ), # END OF TAB PANEL
                  tabPanel("Experiment Info", value = "tab4",
                           sidebarLayout(
                             sidebarPanel(
                               h5("Upload experiment info or upload existing merged data and go to calibration", style="font-weight:bold"),
                               fileInput("expFile", "Select CSV file",
                                         multiple = FALSE,
                                         accept = c("text/csv",
                                                    "text/comma-separated-values,text/plain",
                                                    ".csv")),
                               # Input: Checkbox if file has header ----
                               checkboxInput("header", "Header", TRUE),
                               # Input: Select separator ----
                               radioButtons("sep", "Separator",
                                            choices = c(Comma = ",",
                                                        Semicolon = ";",
                                                        Tab = "\t"),
                                            selected = ","),
                               # Input: Select quotes ----
                               radioButtons("quote", "Quote",
                                            choices = c(None = "",
                                                        "Double Quote" = '"',
                                                        "Single Quote" = "'"),
                                            selected = '"'),  hr(style="border-color: gray"),
                               h5("Select ID columns and merge datasets", style="font-weight:bold"),
                               selectizeInput("mergeIntens", label = "ID Column Features Data", choices="", multiple=TRUE, options=list(placeholder="Load features data")),
                               selectizeInput("mergeExp", label = "ID Column Experiment Info", choices="", multiple=TRUE, options=list(placeholder="Load experiment info")),
                               actionButton("merge", label = "Merge With Features Data"), br(),
                               hr(style="border-color: gray"),
                               h5("Download merged data", style="font-weight:bold"),
                               # actionButton("refreshData2", label = "3) Refresh Data"), br(), br(),
                               downloadButton("downloadData2", "Download Data"), br(),
                               hr(style="border-color: gray"),
                               actionButton("prepare", label = "Prepare Calibration"),
                               hr(style="border-color: gray"),
                               h5("For restart with new data", style="font-weight:bold"),
                               actionButton("deleteData2", label = "Delete Data"), br(),
                             ),
                             mainPanel(
                               DTOutput("experiment")
                             )
                           )
                  ), # END OF TAB PANEL
                  tabPanel("Calibration", value = "tab5",
                           sidebarLayout(
                             sidebarPanel(
                               textInput("folder", "Specify a folder for the analysis results", value=file.path(fs::path_home(), "Documents/IRCApp"), 
                                         placeholder=file.path(fs::path_home(), "Documents/IRCApp")),
                               hr(style="border-color: gray"),
                               #                       h5("Optional: average technical replicates", style="font-weight:bold"),
                               #                       hr(style="border-color: black"),
                               #                       h5("Optional: reshape data from long to wide", style="font-weight:bold"),
                               #                       hr(style="border-color: black"),
                               h5("You can also upload existing data and run the calibration", style="font-weight:bold"),
                               fileInput("prepFile", "Select CSV file",
                                         multiple = FALSE,
                                         accept = c("text/csv",
                                                    "text/comma-separated-values,text/plain",
                                                    ".csv")),
                               hr(style="border-color: gray"),
                               radioButtons("radioPrepro",
                                            label = ("Further preprocessing steps:"),
                                            choices = list("None" = 1,
                                                           "Average technical replicates" = 2,
                                                           "Reshape from long to wide" = 3),
                                            selected = 1),
                               conditionalPanel(
                                 condition = "input.radioPrepro == 2",
                                 hr(style="border-color: gray"),
                                 textInput("combRepsColSI", label = "Column with date:", value = "date"),
                                 numericInput(inputId = "colorsBands",
                                              label = "Number of experiments per date:",
                                              value = 1,
                                              min = 1,
                                              max = 5,
                                              step = 1,
                                              width = NULL
                                 ),
                                 conditionalPanel(
                                   condition = "input.colorsBands > 1",
                                   textInput("combRepsColCL", label = "Column with experiment id:", value = "expID"),
                                 ),
                                 radioButtons("radioReps",
                                              label = ("Choose measure for averaging:"),
                                              choices = list("Mean" = 1,
                                                             "Median" = 2),
                                              selected = 1),
                                 actionButton("combReps", label = "Average Technical Replicates"),
                               ),
                               conditionalPanel(
                                 hr(style="border-color: gray"),
                                 condition = "input.radioPrepro == 3",
                                 textInput("reshapeCol", label = "Column:", value = "Color"),
                                 actionButton("reshapeWide", label = "Reshape"),
                               ),
                               hr(style="border-color: gray"),
                               h5("Download calibration data", style="font-weight:bold"),
                               # actionButton("refreshData3", label = "3) Refresh Data"), br(), br(),
                               downloadButton("downloadData3", "Download Data"),
                               hr(style="border-color: gray"),
                               h5("Calibration", style="font-weight:bold"),
                               textInput("analysisName", label = "Analysis name:", value = "Model1"),
                               radioButtons("chosenModel",
                                            label = "Choose model:",
                                            choices = list("Linear model (lm)" = 1,
                                                           "Local polynomial model (loess)" = 2,
                                                           "Generalized additive model (gam)" = 3),
                                            selected = 1),
                               selectInput("concVar", "Select column with concentration", choices = ""),
                               checkboxInput("useLog", "Logarithmize concentration", value=FALSE),
                               textAreaInput("respVar", label = "Specify the response variable (R expression)"),
                               textAreaInput("subset", label = "Optional: specify subset (logical R expression)"),
                               actionButton("runCali", label = "Run Calibration Analysis"),
                               hr(style="border-color: gray"),
                               h5("For restart with new data", style="font-weight:bold"),
                               actionButton("deleteData3", label = "Delete Data"), br(),
                             ),
                             mainPanel(
                               DTOutput("calibration")
                             )
                           )
                  ), # END OF TAB PANEL
                  tabPanel("Results", value = "tab6",
                           sidebarLayout(
                             sidebarPanel(
                               h4("Open analysis report"),
                               actionButton("openReport", label = "Open")
                             ),
                             mainPanel(
                               h3("Results of Calibration Analysis", style="font-weight:bold"), br(),
                               h4("Calibration model", style="font-weight:bold"),
                               verbatimTextOutput("modelSummary"), br(),
                               plotOutput("plot5"), br(),
                               verbatimTextOutput("LOB"),
                               verbatimTextOutput("LOD"),
                               verbatimTextOutput("LOQ")
                             )
                           )
                  ) # END OF TAB PANEL
      ) # END OF TAB SET PANEL
    )
  )
}
