cal_server <- function( input, output, session ) {
  ###### FIRST TAB
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))
  
  oldopt <- options()
  on.exit(options(oldopt))
  options(shiny.maxRequestSize=100*1024^2) #file can be up to 50 mb; default is 5 mb
  ## initializations
  shinyImageFile <- reactiveValues(shiny_img_origin = NULL, shiny_img_cropped = NULL,
                                   shiny_img_final = NULL, Threshold = NULL)
  IntensData <- NULL
  ExpInfo <- NULL
  MergedData <- NULL
  FILENAME <- NULL
  fit <- NULL
  modelPlot <- NULL
  LOB <- NULL
  LOD <- NULL
  LOQ <- NULL
  predFunc <- NULL
  CalibrationData <- NULL
  
  startAutosave <- reactiveVal(value=FALSE)
  
  observe({recursiveDelete()})
  recursiveDelete <- eventReactive(input$deleteData,{
    isolate({
      IntensData <<- NULL
      output$intens <- renderDT({})
    })
  })
  
  observe({recursiveDelete2()})
  recursiveDelete2 <- eventReactive(input$deleteData2,{
    isolate({
      ExpInfo <<- NULL
      MergedData <<- NULL
      output$experiment <- renderDT({})
    })
  })
  
  observe({recursiveDelete3()})
  recursiveDelete3 <- eventReactive(input$deleteData3,{
    isolate({
      MergedData <<- NULL
      CalibrationData <<- NULL
      output$calibration <- renderDT({})
    })
  })
  
  observe({recursiveRefresh()})
  recursiveRefresh <- eventReactive(input$refreshData,{
    isolate({
      output$intens <- renderDT({
        DF <- IntensData
        datatable(DF)
      })
    })
  })
  
  observe({recursiveRefresh2()})
  recursiveRefresh2 <- eventReactive(input$refreshData2,{
    isolate({
      output$experiment <- renderDT({
        DF <- MergedData
        datatable(DF)
      })
    })
  })
  
  observe({recursiveRefresh3()})
  recursiveRefresh3 <- eventReactive(input$refreshData3,{
    isolate({
      output$calibration <- renderDT({
        DF <- CalibrationData
        datatable(DF)
      })
    })
  })
  
  observe({recursiveExpInfo()})
  
  recursiveExpInfo <- eventReactive(input$expInfo,{
    updateTabsetPanel(session, "tabs", selected = "tab4")
  })
  
  observe({recursiveUploadIntens()})
  recursiveUploadIntens <- eventReactive(input$intensFile,{
    isolate({
      req(input$intensFile)
      tryCatch(
        DF <- read.csv(input$intensFile$datapath, header = TRUE,
                       check.names = FALSE),
        error = function(e){stop(safeError(e))}
      )
      IntensData <<- DF
      output$intens <- renderDT({
        datatable(DF)
      })
      updateSelectizeInput(session, "mergeIntens", choice=colnames(IntensData), options=list(placeholder="Select columns"))
    })
  })
  
  observe({recursiveUploadExpFile()})
  recursiveUploadExpFile <- eventReactive(input$expFile,{
    isolate({
      req(input$expFile)
      tryCatch(
        DF <- read.csv(input$expFile$datapath, header = TRUE,
                       check.names = FALSE),
        error = function(e){stop(safeError(e))}
      )
      ExpInfo <<- DF
      MergedData <<- DF
      suppressWarnings(rm(CalibrationData, pos = 1))
      output$calibration <- renderDT({})
      
      output$experiment <- renderDT({
        datatable(DF)
      })
      updateSelectizeInput(session, "mergeExp", choice=colnames(ExpInfo), options=list(placeholder="Select columns"))
    })
  })
  
  observe({recursiveUploadPrepFile()})
  recursiveUploadPrepFile <- eventReactive(input$prepFile,{
    isolate({
      req(input$prepFile)
      tryCatch(
        DF <- read.csv(input$prepFile$datapath, header = TRUE,
                       check.names = FALSE),
        error = function(e){stop(safeError(e))}
      )
      CalibrationData <<- DF
      MergedData <<- DF
      output$calibration <- renderDT({
        datatable(DF)
      })
      updateSelectInput(session = session, "concVar", choices = names(DF))
    })
  })
  
  observe({recursiveMerge()})
  recursiveMerge <- eventReactive(input$merge,{
    isolate({
      if (is.null(ExpInfo)) {
        showNotification("Experiment info not found.", duration=3, type="error")
      } else if (is.null(IntensData)) {
        showNotification("Intensity data not found.", duration=3, type="error")
      } else if (inherits(try(merge(IntensData, ExpInfo,
                                    by.x = input$mergeIntens,
                                    by.y = input$mergeExp), silent = TRUE), "try-error")) {
        showNotification("Error in the column IDs.", duration=5, type="error")
      } else {
        DF <- merge(IntensData, ExpInfo,
                    by.x = input$mergeIntens,
                    by.y = input$mergeExp)
        
        MergedData <<- DF
        CalibrationData <<- DF
        
        output$experiment <- renderDT({
          datatable(DF)
        })
      }
    })
  })
  
  observe({recursivePrepare()})
  recursivePrepare <- eventReactive(input$prepare,{
    DF <- MergedData
    CalibrationData <<- DF
    
    output$calibration <- renderDT({
      datatable(DF)
    })
    updateSelectInput(session, "concVar", choices = names(DF))
    updateTabsetPanel(session, "tabs", selected = "tab5")
  })
  
  observe({recursiveCombReps()})
  recursiveCombReps <- eventReactive(input$combReps,{
    isolate({
      Cols <- c(grep("shift", colnames(MergedData)),
                grep("height", colnames(MergedData)))
      RES <- NULL
      if(input$colorsBands > 1){
        DF <- MergedData[,c(input$combRepsColSI, input$combRepsColCL)]
        DFuni <- DF[!duplicated(DF),]
        for (i in 1:nrow(DFuni)) {
          sel <- DF[,1] == DFuni[i,1] & DF[,2] == DFuni[i,2]
          tmp <- MergedData[sel, ]
          tmp2 <- tmp[1, ]
          if (input$radioReps == 1) #mean
            tmp2[, Cols] <- colMeans(tmp[, Cols], na.rm = TRUE)
          if (input$radioReps == 2) #median
            tmp2[, Cols] <- apply(tmp[, Cols], 2, median, na.rm = TRUE)
          RES <- rbind(RES, tmp2)
        }
      }else{
        DF <- MergedData[,input$combRepsColSI]
        for (spl in unique(MergedData[, input$combRepsColSI])) {
          tmp <- MergedData[DF == spl, ]
          tmp2 <- tmp[1, ]
          if (input$radioReps == 1) #mean
            tmp2[, Cols] <- colMeans(tmp[, Cols], na.rm = TRUE)
          if (input$radioReps == 2) #median
            tmp2[, Cols] <- apply(tmp[, Cols], 2, median, na.rm = TRUE)
          RES <- rbind(RES, tmp2)
        }
      }
      rownames(RES) <- 1:nrow(RES)
      RES <- RES[order(RES[,input$combRepsColSI]),]
      CalibrationData <<- RES
      
      output$calibration <- renderDT({
        datatable(RES)
      })
    })
  })
  
  observe({recursiveReshapeWide()})
  
  recursiveReshapeWide <- eventReactive(input$reshapeWide,{
    isolate({
      rm.file <- (colnames(CalibrationData) != colnames(MergedData)[1] &
                    colnames(CalibrationData) != input$reshapeCol)
      DF.split <- split(CalibrationData[,rm.file], CalibrationData[,input$reshapeCol])
      
      N <- length(unique(CalibrationData[,input$reshapeCol]))
      if(N > 1){
        DF <- DF.split[[1]]
        Cols <- c(grep("Mean", colnames(DF)),
                  grep("Median", colnames(DF)))
        Cols <- c(Cols, which(colnames(DF) == input$combRepsColSI))
        for(i in 2:N){
          DF <- merge(DF, DF.split[[i]][,Cols], by = input$combRepsColSI,
                      suffixes = paste0(".", names(DF.split)[c(i-1,i)]))
        }
        CalibrationData <<- DF
      }else{
        DF <- CalibrationData
      }
      
      output$calibration <- renderDT({
        datatable(DF)
      })
    })
  })
  
  MODELNUM <- 1
  
  observe({recursiveRunCali()})
  
  recursiveRunCali <- eventReactive(input$runCali,{
    isolate({
      # flush the output and plots
      output$LOB <- renderText({})
      output$LOD <- renderText({})
      output$LOQ <- renderText({})
      output$plot5 <- renderPlot({})
      
      
      PATH.OUT <- input$folder
      if (!file.exists(PATH.OUT)) dir.create(PATH.OUT)
      
      concVar <- input$concVar
      respVar <- paste0("(",input$respVar,")")
      
      if(input$useLog){
        if(input$chosenModel == 3){
          k <- ceiling(length(unique(CalibrationData[,concVar]))/2)
          FORMULA <- paste0(respVar, " ~ s(log10(", concVar, "), k = ", k, ")")  
        }else{
          FORMULA <- paste0(respVar, " ~ log10(", concVar, ")")  
        }
      }else{
        if(input$chosenModel == 3){
          k <- ceiling(length(unique(CalibrationData[,concVar]))/2)
          FORMULA <- paste0(respVar, " ~ s(", concVar, ", k = ", k, ")")  
        }else{
          FORMULA <- paste0(respVar, " ~ ", concVar)
        }
      }
      
      if(input$chosenModel == 1 && !inherits(try(lm(as.formula(FORMULA), data=CalibrationData), silent = TRUE), "try-error")){
        modelName <- "lm"
      } else if(input$chosenModel == 2 && !inherits(try(loess(as.formula(FORMULA), data = CalibrationData), silent = TRUE), "try-error")){
        modelName <- "loess"
      } else if(input$chosenModel == 3 && !inherits(try(gam(as.formula(FORMULA), data = CalibrationData), silent = TRUE), "try-error")){
        modelName <- "gam"
      } else {
        output$modelSummary <- renderPrint({print("Calibration can not be performed. Please check the formula.");
          print(paste0("Formula: ",FORMULA))})
        showNotification("Error in the formula!", duration = 5, type="error")
        updateTabsetPanel(session, "tabs", selected = "tab6")
        return(NULL)
      }
      
      info <- showNotification(paste("Fitting the model..."), duration = 0, type="message")
      
      SUBSET <- input$subset
      
      FILENAME <<- paste0(format(Sys.time(), "%Y%m%d_%H%M%S_"), input$analysisName)
      
      save(CalibrationData, FORMULA, SUBSET, PATH.OUT,
           file = paste0(PATH.OUT,"/", FILENAME, "_Data.RData"))
      if (input$chosenModel == 1) {
        file.copy(from = system.file("markdown", "CalibrationAnalysis(lm).Rmd",
                                     package = "LFApp"),
                  to = paste0(PATH.OUT, "/", FILENAME, "_Analysis.Rmd"))
      } else if (input$chosenModel == 2) {
        file.copy(from = system.file("markdown", "CalibrationAnalysis(loess).Rmd",
                                     package = "LFApp"),
                  to = paste0(PATH.OUT, "/", FILENAME, "_Analysis.Rmd"))
      } else if (input$chosenModel == 3) {
        file.copy(from = system.file("markdown", "CalibrationAnalysis(gam).Rmd",
                                     package = "LFApp"),
                  to = paste0(PATH.OUT, "/", FILENAME, "_Analysis.Rmd"))
      }
      
      rmarkdown::render(input = paste0(PATH.OUT, "/", FILENAME, "_Analysis.Rmd"),
                        output_file = paste0(PATH.OUT, "/", FILENAME, "_Analysis.html"))
      
      # load(file = paste0(PATH.OUT, "/", FILENAME, "Results.RData")) # This line is not necessary, because the parameters are still loaded in the environment.
      
      output$modelSummary <- renderPrint({ fit })
      
      output$plot5 <- renderPlot({
        modelPlot
      })
      output$LOB <- renderText({
        paste0("Limit of Blank (LOB): ", signif(LOB, 3))
      })
      output$LOD <- renderText({
        paste0("Limit of Detection (LOD): ", signif(LOD, 3))
      })
      output$LOQ <- renderText({
        paste0("Limit of Quantification (LOQ): ", signif(LOQ, 3))
      })
      
      # Adding the analysis name and model formula to the table
      # modelName <- rep(modelName, nrow(CalibrationData))
      # modelFormula <- rep(FORMULA, nrow(CalibrationData))
      # modelDF <- cbind(modelName, modelFormula, predFunc(CalibrationData))
      # colnames(modelDF) <- c(paste0(input$analysisName, ".model"), 
      #                        paste0(input$analysisName, ".formula"), 
      #                        paste0(input$analysisName, ".", input$concVar, ".fit"))
      # if(SUBSET != ""){
      #   subsetIndex <- function (x, subset){
      #     e <- substitute(subset)
      #     r <- eval(e, x, parent.frame())
      #     r & !is.na(r)
      #   }
      #   Index <- eval(call("subsetIndex", x = CalibrationData, 
      #                      subset = parse(text = SUBSET)))
      #   modelDF[!Index,] <- NA
      # }
      # DF <- cbind(CalibrationData, modelDF)
      # CalibrationData <<- DF
      # output$calibration <- renderDT({
      #   datatable(DF)
      # })
      
      MODELNUM <<- MODELNUM + 1
      
      updateTextInput(session=session, inputId="analysisName", value=paste0("Model", MODELNUM))
      
      removeNotification(info)
      
      updateTabsetPanel(session, "tabs", selected = "tab6")
    })
  })
  
  observe(resetFolder())
  
  resetFolder <- eventReactive(input$folder,{
    isolate({
      if(substring(input$folder,1,nchar(file.path(fs::path_home()))) != file.path(fs::path_home()))
        updateTextInput(session=session, inputId = "folder", value = file.path(fs::path_home())) 
    })
  })
  
  observe({recursiveOpenReport()})
  
  recursiveOpenReport <- eventReactive(input$openReport,{
    isolate({
      browseURL(paste0(input$folder, "/", FILENAME, "_Analysis.html"),
                browser = getOption("browser"))
    })
  })
  
  #creates the textbox below plot2 about the plot_brush details and etc
  output$thresh <- renderText({
    if(!is.null(shinyImageFile$Threshold))
      paste0("Threshold(s): ", paste0(signif(shinyImageFile$Threshold, 4), collapse = ", "))
  })
  output$meanIntens <- renderText({
    if(!is.null(shinyImageFile$Threshold))
      paste0("Mean intensities: ", paste0(signif(shinyImageFile$Mean_Intensities, 4), collapse = ", "))
  })
  output$medianIntens <- renderText({
    if(!is.null(shinyImageFile$Threshold))
      paste0("Median intensities: ", paste0(signif(shinyImageFile$Median_Intensities, 4), collapse = ", "))
  })
  output$intens <- renderDT({
    DF <- IntensData
    datatable(DF)
  })
  output$folder <- renderPrint({
    paste0("Folder for Results: ", parseDirPath(c(wd=fs::path_home()), input$folder))
  })
  
  #allows user to download data
  output$downloadData2 <- downloadHandler(
    filename = "MergedData.csv",
    content = function(file) {
      write.csv(MergedData, file, row.names = FALSE)
    }
  )
  output$downloadData3 <- downloadHandler(
    filename = "CalibrationData.csv",
    content = function(file) {
      write.csv(CalibrationData, file, row.names = FALSE)
    }
  )
  
  #When user clicks the return to command line button
  #stops the shiny app
  # prevents user from quitting shiny using ^C on commandline
  observe({recursiveStop()})
  
  recursiveStop <- eventReactive(input$stop,{
    isolate({
      suppressWarnings(rm(IntensData, pos = 1))
      suppressWarnings(rm(ExpInfo, pos = 1))
      suppressWarnings(rm(MergedData, pos = 1))
      suppressWarnings(rm(CalibrationData, pos = 1))
      stopApp()
    })
  })
  
  # Checking if workspace file exist
  # observe({
  #   if (file.exists(file.path(fs::path_home(), "Documents/LFApp/cal_autosave.RData"))) {
  #     showModal(modalDialog(
  #       title = "Old workspace backup found",
  #       "Do you want to restore previous workspace?",
  #       h6("If you do not restore the old workspace, it will be overwritten!"),
  #       footer = tagList(
  #         actionButton("no_restore", "No"),
  #         actionButton("restore_work", "Yes")
  #       )
  #     ))
  #   } else {
  #     startAutosave(TRUE)
  #   }
  # })
  # 
  # observeEvent(input$no_restore, {
  #   startAutosave(TRUE)
  #   removeModal()
  # })
  # 
  # observeEvent(input$restore_work, {
  #   load(file=file.path(fs::path_home(), "Documents/LFApp/cal_autosave.RData"))
  #   
  #   # Loading the variables properly. Without this the variables are loaded in the observe scope only
  #   shinyImageFile <<- shinyImageFile
  #   IntensData <<- IntensData
  #   ExpInfo <<- ExpInfo
  #   MergedData <<- MergedData
  #   fit <<- fit
  #   modelPlot <<- modelPlot
  #   LOB <<- LOB
  #   LOD <<- LOD
  #   LOQ <<- LOQ
  #   
  #   # Loading the image on the first plot
  #   if (!is.null(shinyImageFile$shiny_img_final))
  #     output$plot1 <- renderPlot({EBImage::display(shinyImageFile$shiny_img_final, method = "raster")})
  #   
  #   # Loading datatables
  #   output$intens <- renderDT(datatable(IntensData))
  #   output$experiment <- renderDT(datatable(ExpInfo))
  #   output$calibration <- renderDT(datatable(MergedData))
  #   
  #   # Loading model in results tab
  #   if (!is.null(fit) && !is.null(LOB) && !is.null(LOD) && !is.null(LOQ)) {
  #     output$modelSummary <- renderPrint({ fit })
  #     output$plot5 <- renderPlot({ modelPlot })
  #     output$LOB <- renderText({ paste0("Limit of Blank (LOB): ", signif(LOB, 3)) })
  #     output$LOD <- renderText({ paste0("Limit of Detection (LOD): ", signif(LOD, 3)) })
  #     output$LOQ <- renderText({ paste0("Limit of Quantification (LOQ): ", signif(LOQ, 3)) })
  #   }
  #   startAutosave(TRUE)
  #   removeModal()
  # })
  # 
  # # Autosaving every minute
  # observe({
  #   if (startAutosave()) {
  #     invalidateLater(300000, session)
  #     if (!file.exists(file.path(fs::path_home(), "Documents/LFApp"))) dir.create(file.path(fs::path_home(), "Documents/LFApp"))
  #     save(shinyImageFile, IntensData, 
  #          ExpInfo, MergedData, fit, 
  #          modelPlot, LOB, LOD, LOQ,
  #          file=file.path(fs::path_home(), "Documents/LFApp/cal_autosave.RData"))
  #     showNotification("Workspace saved", duration=2, type="message")
  #   }
  # })
  
  # A function to remove the autosave file if the app was closed properly
  # onStop(function() file.remove(file.path(fs::path_home(), "Documents/LFApp/cal_autosave.RData")))
}
