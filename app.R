library(shiny)
library(reticulate)

use_python("python.exe", required = TRUE)
source_python("VAE_GAN_model.py")
columns <- get_column_lists("NP-PC Database(Part).xlsx")

ui <- fluidPage(
  titlePanel("VAE-GAN Gene Expression Predictor"),

  sidebarLayout(
    sidebarPanel(
      checkboxGroupInput("selected_conditions", "Condition Columns:", choices = columns$condition_cols),
      checkboxGroupInput("selected_genes", "Gene Columns:", choices = columns$gene_cols),
      actionButton("run", "Run Prediction")
    ),

    mainPanel(
      tableOutput("prediction")
    )
  )
)

server <- function(input, output) {
  results <- eventReactive(input$run, {
    if (length(input$selected_conditions) == 0 || length(input$selected_genes) == 0) return(NULL)
    run_vae_gan(input$selected_conditions, input$selected_genes)
  })

  output$prediction <- renderTable({
    results()
  })
}

shinyApp(ui = ui, server = server)
