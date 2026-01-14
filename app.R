# ============================================================================
# APPLICATION SHINY - PROJET INONDATIONS (3 MODÈLES)
# Projet Data Science DaMS4 - Polytech Montpellier
# Janvier 2026
# ============================================================================

library(shiny)
library(shinydashboard)
library(DT)
library(plotly)
library(dplyr)
library(ggplot2)
library(MASS)
library(randomForest)
library(RColorBrewer)
library(glmnet)
library(caret)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

charger_donnees <- function() {
  
  # Vérifier si les fichiers existent
  if (file.exists("data/extracted.csv") && file.exists("data/data_Y.csv")) {
    
    cat("Chargement des données réelles...\n")
    
    # Charger X (variables explicatives/entrées) - 44 colonnes
    X <- read.csv("data/extracted.csv", stringsAsFactors = FALSE)
    cat("✓ extracted.csv chargé:", nrow(X), "lignes x", ncol(X), "colonnes\n")
    
    # Charger Y (variables à prédire/sorties) - SMax
    Y <- read.csv("data/data_Y.csv", stringsAsFactors = FALSE)
    cat("✓ data_Y.csv chargé:", nrow(Y), "lignes x", ncol(Y), "colonnes\n")
    
    # Fusionner X et Y par scenario_id
    # X a "scenario_id", Y a "sc" qui correspondent
    if (nrow(X) == nrow(Y)) {
      # Ajouter SMax à X
      df <- X
      df$SMax <- Y$SMax
      cat("✓ Données combinées:", nrow(df), "lignes x", ncol(df), "colonnes\n")
      cat("✓ Variable cible: SMax (Surface maximale inondée)\n")
    } else {
      stop("ERREUR: X et Y n'ont pas le même nombre de lignes!")
    }
    
  } else {
    # Si les fichiers n'existent pas, utiliser des données d'exemple
    warning("Fichiers extracted.csv ou data_Y.csv introuvables. Utilisation de données d'exemple.")
    df <- generer_donnees_exemple()
  }
  
  return(df)
}

generer_donnees_exemple <- function() {
  set.seed(2026)
  n <- 250
  
  df <- data.frame(
    scenario_id = 1:n,
    hauteur_maree = rnorm(n, mean = 5, sd = 1.5),
    niveau_mer = rnorm(n, mean = 50, sd = 10),
    debit_fleuve = runif(n, min = 100, max = 500),
    precipitation = runif(n, min = 0, max = 100),
    vent_vitesse = runif(n, min = 0, max = 80),
    hauteur_max = abs(rnorm(n, mean = 2.5, sd = 1)),
    surface_max = abs(rnorm(n, mean = 150000, sd = 50000)),
    date_simulation = seq(as.Date("2024-01-01"), by = "day", length.out = n),
    zone = sample(c("Centre-ville", "Zone côtière", "Zone rurale"), n, replace = TRUE)
  )
  
  return(df)
}

# Charger les données au démarrage
donnees_brutes <- charger_donnees()

# ============================================================================
# INTERFACE UTILISATEUR (UI)
# ============================================================================

ui <- dashboardPage(
  
  skin = "blue",
  
  dashboardHeader(
    title = "Modélisation des inondations côtières à Gâvres (France)",
    titleWidth = 350
  ),
  
  dashboardSidebar(
    width = 250,
    sidebarMenu(
      menuItem("Accueil", tabName = "accueil"),
      menuItem("Données", tabName = "donnees"),
      menuItem("Modèle 1", tabName = "modele1"),
      menuItem("Modèle 2", tabName = "modele2"),
      menuItem("Modèle 3", tabName = "modele3"),
      menuItem("Comparaison", tabName = "comparaison"),
      menuItem("Simulation SMax", tabName = "Simulation SMax")
    )
  ),
  
  dashboardBody(
    
    tags$head(
      tags$style(HTML("
        .box { border-radius: 5px; }
        .small-box { border-radius: 5px; }
      "))
    ),
    
    tabItems(
      
      # ========== ONGLET ACCUEIL ==========
      tabItem(
        tabName = "accueil",
        
        h2("Projet : Modélisation des Inondations Côtières"),
        
        fluidRow(
          box(
            title = "Contexte du projet",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            h4("Objectif"),
            p("Développer et comparer 3 métamodèles capables de prédire les inondations 
              côtières avec précision tout en réduisant le temps de calcul."),
            hr(),
            h4(" Équipe"),
            p("Groupe : Benali Syrine, Matmar Lysa, Pinta Sébastien, Malek Jamel"),
            hr(),
            h4(" Données"),
            p("Fichiers chargés :"),
            tags$ul(
              tags$li(strong("extracted.csv :"), "250 scénarios avec 44 variables explicatives"),
              tags$li(strong("data_Y.csv :"), "Surface maximale inondée (SMax) pour chaque scénario")
            ),
            tags$p(strong("Variables principales :"), "U (vent), Hs (hauteur vagues), Tp (période vagues), 
                   NM (niveau mer), T (marée), et leurs dérivées statistiques"),
            hr(),
            h4("Objectifs"),
            tags$ol(
              tags$li("Explorer et visualiser les 44 features extraites"),
              tags$li("Implémenter 3 méthodes de modélisation pour prédire SMax"),
              tags$li("Comparer leurs performances (RMSE, R², MAE)"),
              tags$li("Identifier la meilleure approche pour prédire la surface inondée")
            )
          )
        ),
        
        fluidRow(
          valueBoxOutput("vbox_scenarios", width = 4),
          valueBoxOutput("vbox_variables", width = 4),
          valueBoxOutput("vbox_targets", width = 4)
        )
      ),
      
      # ========== ONGLET DONNÉES ==========
      tabItem(
        tabName = "donnees",
        
        h2("Exploration des données"),
        
        fluidRow(
          box(
            title = "Statistiques descriptives",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            DTOutput("table_stats")
          )
        ),
        
        fluidRow(
          box(
            title = "Variables X (Entrées)",
            status = "info",
            solidHeader = TRUE,
            width = 6,
            DTOutput("table_X")
          ),
          box(
            title = "Variables Y (Sorties)",
            status = "success",
            solidHeader = TRUE,
            width = 6,
            DTOutput("table_Y")
          )
        ),
      ),
      
      # ========== ONGLET MODÈLE 1 (RÉGRESSION LINÉAIRE) ==========
      tabItem(
        tabName = "modele1",
        
        h2("Modèle 1 : Régression Linéaire (Backward Selection)"),
        
        
        fluidRow(
          # Boîte de Configuration
          box(
            title = "Configuration",
            status = "primary", solidHeader = TRUE, width = 4,
            tags$ul(
              tags$li("Split : 80% Entraînement / 20% Test"),
              tags$li("Seed : 123 (Fixé)"),
              tags$li("Sélection : Backward AIC"),
              tags$li("Nettoyage : Suppression des constantes & fuites")
            ),
            br(),
            actionButton("run_m1", "Lancer le calcul", class = "btn-primary btn-lg btn-block")
          ),
          box(
            title = "Résultats Clés",
            status = "primary", solidHeader = TRUE, width = 8,
            fluidRow(
              column(5, 
                     h4("Métriques (sur Test)"),
                     tableOutput("metrics_m1")
              ),
              column(7,
                     h4("Variables Retenues"),
                     uiOutput("vars_m1"), 
                     hr()
              )
            )
          )
        ),
        
        fluidRow(
          box(
            title = "1. Observed vs Predicted (Jeu de Test)",
            status = "success", solidHeader = TRUE, width = 8,
            plotOutput("plot_obs_pred", height = "400px")
          ),
        ),
        
        # --- LIGNE 3 : DIAGNOSTICS (RESIDUS & QQ) ---
        fluidRow(
          # Residuals vs Fitted
          box(
            title = "2. Residuals vs Fitted (Global)",
            status = "info", solidHeader = TRUE, width = 6,
            plotOutput("plot_res_fit", height = "350px"),
            footer = "Recherchez une forme d'entonnoir (hétéroscédasticité)."
          ),
          
          # Normal Q-Q Plot
          box(
            title = "3. Normal Q-Q Plot (Global)",
            status = "info", solidHeader = TRUE, width = 6,
            plotOutput("plot_qq", height = "350px"),
            footer = "Les points doivent suivre la ligne rouge (Normalité)."
          )
        ),
        
        # --- LIGNE 4 : LEADERBOARD DES PARAMÈTRES ---
        fluidRow(
          box(
            title = "4. Leaderboard des Paramètres",
            status = "warning", solidHeader = TRUE, width = 12,
            p("Ce graphique montre le poids (coefficient standardisé) de toutes les variables. Les barres vertes sont les 16 variables retenues par le backward selection, les grises sont celles éliminées."),
            plotlyOutput("plot_importance_m1", height = "700px")
          )
        ),
        
        # --- LIGNE 5 : SIMULATEUR DE SENSIBILITÉ ---
        fluidRow(
          box(
            title = "5. Simulateur de Sensibilité",
            status = "success", solidHeader = TRUE, width = 12,
            p("Visualisez comment chaque variable influence la surface inondée (SMax). Les variables marquées ✓ sont celles retenues par le backward selection."),
            fluidRow(
              column(4,
                     selectInput("var_sensitivity_m1", 
                                 "Choisir une variable (✓ = retenue) :", 
                                 choices = NULL)
              ),
              column(8,
                     p(em("La ligne rouge montre la tendance linéaire. Comparez les variables retenues (✓) aux non-retenues pour voir la différence."))
              )
            ),
            plotlyOutput("plot_sensitivity_m1", height = "400px")
          )
        ),
      ),
      
      # ========== ONGLET MODÈLE 2 : LASSO ==========
      tabItem(
        tabName = "modele2",
        
        h2("Modèle 2 : Régression Lasso"),
        # --- LIGNE 2 : CONFIGURATION & MÉTRIQUES ---
        fluidRow(
          box(
            title = "Configuration",
            status = "warning", solidHeader = TRUE, width = 4,
            sliderInput("m2_train_pct", "% Données d'entraînement :",
                        min = 60, max = 90, value = 80, step = 5),
            sliderInput("m2_nfolds", "Nombre de folds (CV) :",
                        min = 3, max = 10, value = 5, step = 1),
            tags$ul(
              tags$li("Standardisation : Oui (centrage-réduction)"),
              tags$li("Seed : 42 (Fixé)"),
              tags$li("Sélection lambda : lambda.min")
            ),
            hr(),
            actionButton("m2_run", "Lancer Lasso CV", 
                         class = "btn-warning btn-lg btn-block")
          ),
          box(
            title = "Résultats Clés",
            status = "warning", solidHeader = TRUE, width = 8,
            fluidRow(
              column(6,
                     h4("Métriques (Jeu de Test)"),
                     tableOutput("m2_metrics")
              ),
              column(6,
                     h4("Paramètres Optimaux"),
                     tableOutput("m2_params")
              )
            ),
            hr(),
            uiOutput("m2_vars_info")
          )
        ),
        
        # --- LIGNE 3 : COURBE MSE CV ---
        fluidRow(
          box(
            title = "1. Courbe MSE - Validation Croisée (5-fold)",
            status = "info", solidHeader = TRUE, width = 12,
            p("Ce graphique montre l'erreur MSE en fonction de lambda. La ligne verticale indique le lambda optimal."),
            plotlyOutput("m2_mse_curve", height = "400px")
          )
        ),
        
        # --- LIGNE 4 : CHEMIN DES COEFFICIENTS ---
        fluidRow(
          box(
            title = "2. Chemin des Coefficients (Lasso Path)",
            status = "primary", solidHeader = TRUE, width = 12,
            p("Ce graphique montre comment les coefficients évoluent avec lambda. Plus lambda augmente, plus les coefficients sont réduits vers zéro."),
            plotlyOutput("m2_coef_path", height = "450px")
          )
        ),
        
        # --- LIGNE 5 : PRÉDICTIONS & RÉSIDUS ---
        fluidRow(
          box(
            title = "3. Prédictions vs Valeurs Observées (Jeu de Test)",
            status = "success", solidHeader = TRUE, width = 6,
            plotlyOutput("m2_pred_plot", height = "400px")
          ),
          box(
            title = "4. Distribution des Résidus",
            status = "success", solidHeader = TRUE, width = 6,
            plotlyOutput("m2_residuals", height = "400px")
          )
        ),
        
        # --- LIGNE 6 : IMPORTANCE DES COEFFICIENTS ---
        fluidRow(
          box(
            title = "5. Coefficients Sélectionnés (Non-nuls)",
            status = "warning", solidHeader = TRUE, width = 12,
            p("Seules les variables avec un coefficient non-nul sont affichées. C'est la force du Lasso : sélection automatique des variables."),
            plotlyOutput("m2_importance", height = "500px")
          )
        ),
      ),
      
      # ========== ONGLET MODÈLE 3 : RANDOM FOREST ==========
      tabItem(
        tabName = "modele3",
        
        h2("Modèle 3 : Random Forest"),
        
        fluidRow(
          box(
            title = "Configuration",
            status = "danger", solidHeader = TRUE, width = 4,
            sliderInput("m3_ntree", "Nombre d'arbres :",
                        min = 100, max = 500, value = 400, step = 50),
            sliderInput("m3_seuil_importance", "Seuil %IncMSE (filtrage variables) :",
                        min = 0, max = 10, value = 1, step = 0.5),
            tags$ul(
              tags$li("Validation : Out-Of-Bag (OOB)"),
              tags$li("Seed : 123 (Fixé)"),
              tags$li("Données : 250 scénarios")
            ),
            hr(),
            actionButton("m3_run", "Lancer Random Forest", 
                         class = "btn-danger btn-lg btn-block")
          ),
          box(
            title = "Résultats Clés",
            status = "danger", solidHeader = TRUE, width = 8,
            fluidRow(
              column(6,
                     h4("Modèle Initial (toutes variables)"),
                     tableOutput("m3_metrics_initial")
              ),
              column(6,
                     h4("Modèle Optimisé (variables filtrées)"),
                     tableOutput("m3_metrics_optimized")
              )
            ),
            hr(),
            uiOutput("m3_vars_info")
          )
        ),
        
        fluidRow(
          box(
            title = "1. Convergence de l'Erreur en Fonction du Nombre d'Arbres",
            status = "info", solidHeader = TRUE, width = 12,
            p("Ce graphique montre que l'erreur se stabilise entre 150-200 arbres. Nous utilisons 400 arbres par sécurité."),
            plotlyOutput("m3_convergence", height = "350px")
          )
        ),
        
        # --- LIGNE 4 : IMPORTANCE DES VARIABLES ---
        fluidRow(
          box(
            title = "2. Importance des Variables (%IncMSE) - Toutes vs Retenues",
            status = "warning", solidHeader = TRUE, width = 12,
            p("Les variables avec %IncMSE < seuil sont considérées comme du bruit statistique. Barres vertes = retenues, grises = éliminées."),
            plotlyOutput("m3_importance", height = "600px")
          )
        ),
        
        fluidRow(
          box(
            title = "3. Prédictions OOB vs Valeurs Observées (Validation Interne)",
            status = "success", solidHeader = TRUE, width = 6,
            p("La validation Out-Of-Bag simule la réponse du modèle face à des scénarios inédits."),
            plotlyOutput("m3_pred_plot", height = "400px")
          ),
          
          box(
            title = "4. Répartition de l'Importance des Variables",
            status = "primary", solidHeader = TRUE, width = 6,
            p("Visualisez comment l'importance se répartit entre les variables."),
            plotlyOutput("m3_importance_pie", height = "350px")
          )
        ),
      ),
      
      # ========== ONGLET COMPARAISON ==========
      tabItem(
        tabName = "comparaison",
        
        h2("Comparaison des 3 méthodes"),
        
        fluidRow(
          box(
            title = "Performance des modèles - R²",
            status = "info",
            solidHeader = TRUE,
            width = 6,
            plotlyOutput("plot_comparison_r2", height = "350px")
          ),
          box(
            title = "Performance des modèles - RMSE",
            status = "info",
            solidHeader = TRUE,
            width = 6,
            plotlyOutput("plot_comparison_rmse", height = "350px")
          )
        ),
        
        fluidRow(
          box(
            title = "Tableau récapitulatif complet",
            status = "warning",
            solidHeader = TRUE,
            width = 8,
            DTOutput("table_comparison")
          ),
          box(
            title = "Conclusion",
            status = "success",
            solidHeader = TRUE,
            width = 4,
            h4("🏆 Meilleure méthode :"),
            verbatimTextOutput("best_method"),
            hr(),
            h4("📊 Points clés :"),
            htmlOutput("key_points")
          )
        ),
        
        fluidRow(
          box(
            title = "Comparaison visuelle des prédictions",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            plotlyOutput("plot_comparison_predictions", height = "400px")
          )
        )
      )
    )
  )
)

# ============================================================================
# LOGIQUE SERVEUR
# ============================================================================

server <- function(input, output, session) {
  
  # --- Données réactives ---
  donnees <- reactive({
    donnees_brutes
  })
  
  # Identifier les colonnes X et Y
  colonnes_info <- reactive({
    df <- donnees()
    
    # Pour vos données réelles
    # Y = SMax (la variable à prédire)
    # X = toutes les autres variables numériques sauf scenario_id
    
    Y_names <- c("SMax")
    
    # Variables X = toutes les variables numériques sauf scenario_id et SMax
    all_numeric <- names(df)[sapply(df, is.numeric)]
    X_names <- setdiff(all_numeric, c("scenario_id", "SMax"))
    
    list(X = X_names, Y = Y_names)
  })
  
  # Mettre à jour les choix de variables
  observe({
    cols <- colonnes_info()
    
    # Pour les analyses
    updateSelectInput(session, "var_x_ana", choices = cols$X)
    updateSelectInput(session, "var_y_ana", choices = cols$Y)
    updateSelectInput(session, "couleur_ana", choices = c("Aucune", names(donnees())))
    
    # Pour la distribution Y
    updateSelectInput(session, "var_y_dist", choices = cols$Y)
    
    # Pour les modèles
    updateSelectInput(session, "m1_target", choices = cols$Y)
    updateSelectInput(session, "m2_target", choices = cols$Y)
    updateSelectInput(session, "m3_target", choices = cols$Y)
    
    updateCheckboxGroupInput(session, "m1_vars", choices = cols$X, selected = cols$X)
  })
  
  # ========== ONGLET ACCUEIL ==========
  
  output$vbox_scenarios <- renderValueBox({
    valueBox(
      value = nrow(donnees()),
      subtitle = "Scénarios d'inondation",
      icon = icon("water"),
      color = "blue"
    )
  })
  
  output$vbox_variables <- renderValueBox({
    cols <- colonnes_info()
    valueBox(
      value = length(cols$X),
      subtitle = "Features extraites (Variables X)",
      icon = icon("list"),
      color = "green"
    )
  })
  
  output$vbox_targets <- renderValueBox({
    df <- donnees()
    moy <- mean(df$SMax, na.rm = TRUE)
    valueBox(
      value = paste(round(moy/1000, 1), "k m²"),
      subtitle = "Surface moyenne inondée",
      icon = icon("water"),
      color = "yellow"
    )
  })
  
  # ========== ONGLET DONNÉES ==========
  
  output$table_stats <- renderDT({
    df <- donnees()
    
    stats <- data.frame(
      Variable = names(df),
      Type = sapply(df, function(x) class(x)[1]),
      N = nrow(df),
      NA_count = sapply(df, function(x) sum(is.na(x))),
      Min = sapply(df, function(x) if(is.numeric(x)) round(min(x, na.rm=TRUE), 2) else NA),
      Moyenne = sapply(df, function(x) if(is.numeric(x)) round(mean(x, na.rm=TRUE), 2) else NA),
      Max = sapply(df, function(x) if(is.numeric(x)) round(max(x, na.rm=TRUE), 2) else NA)
    )
    
    datatable(stats, options = list(pageLength = 15, scrollX = TRUE))
  })
  
  output$table_X <- renderDT({
    cols <- colonnes_info()
    df <- donnees()[, cols$X, drop = FALSE]
    datatable(head(df, 50), options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$table_Y <- renderDT({
    cols <- colonnes_info()
    df <- donnees()[, cols$Y, drop = FALSE]
    datatable(head(df, 50), options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$plot_na <- renderPlotly({
    df <- donnees()
    na_counts <- colSums(is.na(df))
    
    plot_ly(x = names(na_counts), y = na_counts, type = "bar",
            marker = list(color = "#f39c12")) %>%
      layout(title = "Nombre de valeurs manquantes par variable",
             xaxis = list(title = ""),
             yaxis = list(title = "Nombre de NA"))
  })
  
  output$plot_y_dist <- renderPlotly({
    req(input$var_y_dist)
    df <- donnees()
    
    plot_ly(x = df[[input$var_y_dist]], type = "histogram", nbinsx = 30,
            marker = list(color = "#3498db")) %>%
      layout(title = paste("Distribution de", input$var_y_dist),
             xaxis = list(title = input$var_y_dist),
             yaxis = list(title = "Fréquence"))
  })
  
  # ========== MODÈLE 1 ===========
  
  results_m1 <- eventReactive(input$run_m1, {
    
    # ----------------------------
    # 1. CHARGEMENT & FUSION 
    # ----------------------------
    X <- read.csv("data/extracted.csv")
    Y <- read.csv("data/data_Y.csv")
    
    # Jointure par scenario_id = sc
    data_merged <- inner_join(X, Y, by = c("scenario_id" = "sc"))
    
    # ----------------------------
    # 2. NETTOYAGE 
    # ----------------------------
    # Supprimer scenario_id, S_max et S_mean (comme dans le RMarkdown)
    cols_to_remove <- c("scenario_id", "S_max", "S_mean")
    data_prep <- data_merged[, !(names(data_merged) %in% cols_to_remove)]
    
    # Suppression des colonnes constantes (écart-type = 0)
    col_stds <- sapply(data_prep, sd, na.rm = TRUE)
    data_clean <- data_prep[, col_stds > 0]
    
    # S'assurer que SMax est bien présent
    if (!"SMax" %in% names(data_clean)) {
      data_clean$SMax <- data_prep$SMax
    }
    
    # ----------------------------
    # 3. STANDARDISATION 
    # ----------------------------
    data_scaled <- data_clean
    target_idx <- which(names(data_clean) == "SMax")
    # Standardiser toutes les colonnes SAUF SMax
    data_scaled[, -target_idx] <- scale(data_clean[, -target_idx])
    
    # ----------------------------
    # 4. SPLIT 80 / 20 (SEED 123) 
    # ----------------------------
    set.seed(123) 
    index <- sample(1:nrow(data_scaled), round(0.8 * nrow(data_scaled)))
    train_data <- data_scaled[index, ]
    test_data  <- data_scaled[-index, ]
    
    # ----------------------------
    # 5. MODELE + BACKWARD
    # ----------------------------
    model_full <- lm(SMax ~ ., data = train_data)
    model_final <- step(model_full, direction = "backward", trace = 0)
    
    # ----------------------------
    # 6. CALCULS POUR LES METRIQUES 
    # ----------------------------
    test_data$pred <- predict(model_final, newdata = test_data)
    rss <- sum((test_data$pred - test_data$SMax)^2)
    tss <- sum((test_data$SMax - mean(test_data$SMax))^2)
    r2_test <- 1 - (rss/tss)
    
    rmse_test <- sqrt(mean((test_data$SMax - test_data$pred)^2))
    mae_test <- mean(abs(test_data$SMax - test_data$pred))
    
    # ----------------------------
    # 7. CALCULS POUR LES GRAPHES 2 & 3 
    # ----------------------------
    data_scaled$pred <- predict(model_final, newdata = data_scaled)
    data_scaled$residus <- data_scaled$SMax - data_scaled$pred
    
    # Récupérer les variables retenues par le modèle
    vars_retenues <- names(coef(model_final))[-1]  # Sans l'intercept
    
    # Récupérer toutes les variables (modèle complet)
    all_vars <- names(coef(model_full))[-1]  # Sans l'intercept
    
    list(
      model = model_final,
      model_full = model_full,  # Modèle complet pour comparaison
      r2 = r2_test,
      rmse = rmse_test,
      mae = mae_test,
      # Données pour Graphe 1 
      obs_test = test_data$SMax,
      pred_test = test_data$pred,
      # Données pour Graphes 2 & 3 
      pred_full = data_scaled$pred,
      residus_full = data_scaled$residus,
      # Données pour graphiques interactifs
      data_scaled = data_scaled,
      vars_retenues = vars_retenues,
      all_vars = all_vars  # Toutes les variables
    )
  })
  
  # ==========================================================================
  # OUTPUTS
  # ==========================================================================
  
  output$metrics_m1 <- renderTable({
    res <- results_m1()
    data.frame(
      Métrique = c("R² (Test)", "RMSE", "MAE"),
      Valeur = c(round(res$r2, 4), round(res$rmse, 0), round(res$mae, 0))
    )
  })
  
  output$vars_m1 <- renderUI({
    res <- results_m1()
    vars <- names(coef(res$model))[-1]
    HTML(paste0("<b>Variables retenues (", length(vars), ") :</b><br>", paste(vars, collapse = ", ")))
  })
  
  # GRAPHE 1 : Obs vs Pred 
  output$plot_obs_pred <- renderPlot({
    res <- results_m1()
    ggplot(data.frame(Obs = res$obs_test, Pred = res$pred_test), aes(Obs, Pred)) +
      geom_point(alpha = 0.7, color = "steelblue") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      labs(title = paste("Observed vs Predicted (Test) – R² =", round(res$r2, 3)),
           x = "Réalité (Test)", y = "Prédiction (Test)") +
      theme_minimal()
  })
  
  # GRAPHE 2 : Residuals vs Fitted 
  output$plot_res_fit <- renderPlot({
    res <- results_m1()
    ggplot(data.frame(Fitted = res$pred_full, Res = res$residus_full), aes(Fitted, Res)) +
      geom_point(alpha = 0.5, color = "darkgreen") +
      geom_hline(yintercept = 0, color = "red") +
      geom_smooth(method = "loess", se = FALSE, color = "orange") +
      labs(title = "Residuals vs Fitted (Global)", x = "Valeurs Prédites", y = "Résidus") +
      theme_minimal()
  })
  
  # GRAPHE 3 : QQ Plot 
  output$plot_qq <- renderPlot({
    res <- results_m1()
    ggplot(data.frame(res = res$residus_full), aes(sample = res)) +
      stat_qq(color = "purple", alpha = 0.5) +
      stat_qq_line(color = "red") +
      labs(title = "Normal Q-Q Plot (Global)", x = "Théorique", y = "Echantillon") +
      theme_minimal()
  })
  
  # GRAPHE 4 : Leaderboard des Paramètres (Importance) - TOUTES LES VARIABLES
  output$plot_importance_m1 <- renderPlotly({
    res <- results_m1()
    
    # Extraire les coefficients du modèle COMPLET (toutes les 44 variables)
    coef_full <- as.data.frame(summary(res$model_full)$coefficients)
    coef_full$Variable <- rownames(coef_full)
    coef_full <- coef_full[-1, ]  # Enlever l'intercept
    
    # Marquer les variables retenues par le backward selection
    coef_full$Retenue <- ifelse(coef_full$Variable %in% res$vars_retenues, "Retenue (16)", "Non retenue")
    
    # Créer le graphique
    p <- ggplot(coef_full, aes(x = reorder(Variable, abs(Estimate)), 
                               y = Estimate, 
                               fill = Retenue,
                               text = paste("Variable:", Variable, 
                                            "<br>Coefficient:", round(Estimate, 0),
                                            "<br>p-value:", format.pval(`Pr(>|t|)`, digits = 3),
                                            "<br>Statut:", Retenue))) +
      geom_bar(stat = "identity") +
      coord_flip() +
      scale_fill_manual(values = c("Retenue (16)" = "#27ae60", "Non retenue" = "#bdc3c7"), 
                        name = "Sélection Backward") +
      labs(title = paste("Importance des Variables - TOUTES (", nrow(coef_full), ") vs Retenues (", length(res$vars_retenues), ")"),
           x = "Variables", 
           y = "Coefficient (standardisé)") +
      theme_minimal() +
      theme(legend.position = "bottom",
            axis.text.y = element_text(size = 7))
    
    ggplotly(p, tooltip = "text") %>%
      layout(hoverlabel = list(bgcolor = "white"),
             height = 700)  # Plus grand pour voir toutes les variables
  })
  
  # Mettre à jour les choix du selectInput pour la sensibilité - TOUTES les variables
  observe({
    res <- tryCatch(results_m1(), error = function(e) NULL)
    if (!is.null(res)) {
      # Créer des choix avec labels pour distinguer les variables retenues
      all_choices <- res$all_vars
      names(all_choices) <- ifelse(all_choices %in% res$vars_retenues, 
                                   paste(all_choices, "✓"), 
                                   all_choices)
      updateSelectInput(session, "var_sensitivity_m1", 
                        choices = all_choices,
                        selected = res$vars_retenues[1])
    }
  })
  
  # GRAPHE 5 : Simulateur de Sensibilité
  output$plot_sensitivity_m1 <- renderPlotly({
    req(input$var_sensitivity_m1)
    res <- results_m1()
    
    df <- res$data_scaled
    var_choice <- input$var_sensitivity_m1
    
    # Vérifier que la variable existe
    if (!(var_choice %in% names(df))) {
      return(plot_ly() %>% layout(title = "Variable non disponible"))
    }
    
    # Créer le graphique de sensibilité
    p <- ggplot(df, aes_string(x = var_choice, y = "SMax")) +
      geom_point(alpha = 0.4, color = "steelblue", size = 2) +
      geom_smooth(method = "lm", color = "red", se = TRUE, alpha = 0.2) +
      labs(title = paste("Impact de", var_choice, "sur la Surface Inondée (SMax)"),
           x = paste(var_choice, "(standardisé)"),
           y = "Surface Inondée (m²)") +
      theme_minimal()
    
    ggplotly(p) %>%
      layout(hoverlabel = list(bgcolor = "white"))
  })
  
  # Mettre à jour les choix pour le graphe 3D
  observe({
    res <- tryCatch(results_m1(), error = function(e) NULL)
    if (!is.null(res)) {
      all_choices <- res$all_vars
      # Définir des valeurs par défaut intelligentes
      default_x <- if("NMmax" %in% all_choices) "NMmax" else all_choices[1]
      default_y <- if("Hs_max" %in% all_choices) "Hs_max" else all_choices[2]
      default_z <- "SMax"
      
      updateSelectInput(session, "var_3d_x", choices = all_choices, selected = default_x)
      updateSelectInput(session, "var_3d_y", choices = all_choices, selected = default_y)
      updateSelectInput(session, "var_3d_z", choices = c("SMax", all_choices), selected = default_z)
    }
  })
  
  # GRAPHE 6 : Scatter Plot 3D Interactif
  output$plot_3d_scatter <- renderPlotly({
    req(input$var_3d_x, input$var_3d_y, input$var_3d_z)
    res <- results_m1()
    
    df <- res$data_scaled
    x_var <- input$var_3d_x
    y_var <- input$var_3d_y
    z_var <- input$var_3d_z
    
    # Vérifier que les variables existent
    if (!(x_var %in% names(df)) || !(y_var %in% names(df)) || !(z_var %in% names(df))) {
      return(plot_ly() %>% layout(title = "Variables non disponibles"))
    }
    
    # Créer le texte pour le hover
    hover_text <- paste(
      "Scénario:", 1:nrow(df),
      "<br>", x_var, ":", round(df[[x_var]], 2),
      "<br>", y_var, ":", round(df[[y_var]], 2),
      "<br>", z_var, ":", round(df[[z_var]], 0)
    )
    
    # Créer le graphique 3D
    plot_ly(df, 
            x = ~get(x_var), 
            y = ~get(y_var), 
            z = ~get(z_var),
            type = "scatter3d", 
            mode = "markers",
            marker = list(
              size = 5,
              color = ~get(z_var),
              colorscale = "Viridis",
              showscale = TRUE,
              colorbar = list(title = z_var)
            ),
            text = hover_text,
            hoverinfo = "text") %>%
      layout(
        title = paste("Relation 3D:", x_var, "vs", y_var, "vs", z_var),
        scene = list(
          xaxis = list(title = x_var),
          yaxis = list(title = y_var),
          zaxis = list(title = z_var),
          camera = list(
            eye = list(x = 1.5, y = 1.5, z = 1.2)
          )
        ),
        hoverlabel = list(bgcolor = "white")
      )
  })
  
  # ========== MODÈLE 2 : LASSO ==========
  
  resultats_m2 <- eventReactive(input$m2_run, {
    
    # ----------------------------
    # 1. CHARGEMENT & PRÉPARATION
    # ----------------------------
    X <- read.csv("data/extracted.csv")
    Y <- read.csv("data/data_Y.csv")
    
    # Harmonisation du nom de colonne
    if ("sc" %in% colnames(Y)) {
      colnames(Y)[colnames(Y) == "sc"] <- "scenario_id"
    }
    
    # Fusion
    data_merged <- inner_join(X, Y, by = "scenario_id")
    
    # Préparation X / y
    target_col <- "SMax"
    cols_to_remove <- c("scenario_id", target_col, "S_max", "S_mean")
    cols_to_remove <- cols_to_remove[cols_to_remove %in% names(data_merged)]
    X_data <- data_merged[, !(names(data_merged) %in% cols_to_remove)]
    y_data <- data_merged[[target_col]]
    
    # ----------------------------
    # 2. TRAIN / TEST SPLIT
    # ----------------------------
    set.seed(42)
    train_pct <- input$m2_train_pct / 100
    train_index <- createDataPartition(y_data, p = train_pct, list = FALSE)
    
    X_train <- X_data[train_index, ]
    X_test  <- X_data[-train_index, ]
    y_train <- y_data[train_index]
    y_test  <- y_data[-train_index]
    
    # ----------------------------
    # 3. SCALING (centrage-réduction)
    # ----------------------------
    scaler <- preProcess(X_train, method = c("center", "scale"))
    X_train_scaled <- predict(scaler, X_train)
    X_test_scaled  <- predict(scaler, X_test)
    X_full_scaled  <- predict(scaler, X_data)
    
    # ----------------------------
    # 4. LASSO CV
    # ----------------------------
    set.seed(42)
    lasso_cv <- cv.glmnet(
      x = as.matrix(X_train_scaled),
      y = y_train,
      alpha = 1,
      nfolds = input$m2_nfolds,
      standardize = FALSE,
      keep = TRUE
    )
    
    best_lambda <- lasso_cv$lambda.min
    
    # ----------------------------
    # 5. PRÉDICTIONS & MÉTRIQUES
    # ----------------------------
    y_pred_test <- as.vector(predict(lasso_cv, s = best_lambda, newx = as.matrix(X_test_scaled)))
    y_pred_full <- as.vector(predict(lasso_cv, s = best_lambda, newx = as.matrix(X_full_scaled)))
    
    # R² sur jeu de test (comme dans le RMarkdown: cor()^2)
    r2_test <- cor(y_test, y_pred_test)^2
    
    # R² sur données complètes
    r2_full <- cor(y_data, y_pred_full)^2
    
    rmse_test <- sqrt(mean((y_test - y_pred_test)^2))
    mae_test <- mean(abs(y_test - y_pred_test))
    
    # ----------------------------
    # 6. COEFFICIENTS SÉLECTIONNÉS
    # ----------------------------
    coef_lasso <- coef(lasso_cv, s = best_lambda)
    coef_df <- data.frame(
      feature = rownames(coef_lasso),
      coefficient = as.numeric(coef_lasso)
    ) %>%
      filter(coefficient != 0, feature != "(Intercept)") %>%
      arrange(desc(abs(coefficient)))
    
    # ----------------------------
    # 7. LASSO PATH (pour graphique)
    # ----------------------------
    lasso_path <- glmnet(
      x = as.matrix(X_train_scaled),
      y = y_train,
      alpha = 1,
      standardize = FALSE
    )
    
    list(
      lasso_cv = lasso_cv,
      lasso_path = lasso_path,
      best_lambda = best_lambda,
      r2 = r2_test,
      r2_full = r2_full,
      rmse = rmse_test,
      mae = mae_test,
      y_test = y_test,
      y_pred_test = y_pred_test,
      y_full = y_data,
      y_pred_full = y_pred_full,
      coef_df = coef_df,
      n_selected = nrow(coef_df),
      X_train_scaled = X_train_scaled,
      X_test_scaled = X_test_scaled,
      X_full_scaled = X_full_scaled,
      all_features = names(X_data)
    )
  })
  
  # --- MÉTRIQUES ---
  output$m2_metrics <- renderTable({
    res <- resultats_m2()
    data.frame(
      Métrique = c("R² (Test)", "R² (Full)", "RMSE", "MAE"),
      Valeur = c(
        round(res$r2, 4),
        round(res$r2_full, 4),
        round(res$rmse, 0),
        round(res$mae, 0)
      )
    )
  })
  
  # --- PARAMÈTRES ---
  output$m2_params <- renderTable({
    res <- resultats_m2()
    data.frame(
      Paramètre = c("Lambda optimal", "Nb variables sélectionnées"),
      Valeur = c(
        format(res$best_lambda, scientific = TRUE, digits = 4),
        res$n_selected
      )
    )
  })
  
  # --- INFO VARIABLES ---
  output$m2_vars_info <- renderUI({
    res <- resultats_m2()
    all_selected_vars <- res$coef_df$feature
    HTML(paste0(
      "<p><b>Variables sélectionnées (", res$n_selected, ") :</b> ",
      paste(all_selected_vars, collapse = ", "),
      "</p>",
      "<p><b>Variables éliminées :</b> ", length(res$all_features) - res$n_selected, " (coefficient = 0)</p>"
    ))
  })
  
  # --- GRAPHE 1 : COURBE MSE CV ---
  output$m2_mse_curve <- renderPlotly({
    res <- resultats_m2()
    lasso_cv <- res$lasso_cv
    
    # Préparer les données
    df_mse <- data.frame(
      lambda = lasso_cv$lambda,
      mse = lasso_cv$cvm,
      mse_upper = lasso_cv$cvup,
      mse_lower = lasso_cv$cvlo
    )
    
    plot_ly() %>%
      # Bande d'écart-type
      add_ribbons(data = df_mse, x = ~lambda, ymin = ~mse_lower, ymax = ~mse_upper,
                  fillcolor = "rgba(241, 196, 15, 0.3)", line = list(color = "transparent"),
                  name = "± 1 SE", showlegend = TRUE) %>%
      # Ligne MSE moyenne
      add_trace(data = df_mse, x = ~lambda, y = ~mse, type = "scatter", mode = "lines",
                line = list(color = "#e74c3c", width = 2),
                name = "MSE moyen") %>%
      # Ligne lambda optimal
      add_trace(x = c(res$best_lambda, res$best_lambda), y = c(min(df_mse$mse_lower), max(df_mse$mse_upper)),
                type = "scatter", mode = "lines",
                line = list(color = "#27ae60", dash = "dash", width = 2),
                name = paste("Lambda optimal:", format(res$best_lambda, scientific = TRUE, digits = 3))) %>%
      layout(title = "Courbe MSE - Validation Croisée",
             xaxis = list(title = "Lambda (λ)", type = "log"),
             yaxis = list(title = "Mean Squared Error (MSE)"),
             showlegend = TRUE)
  })
  
  # --- GRAPHE 2 : CHEMIN DES COEFFICIENTS ---
  output$m2_coef_path <- renderPlotly({
    res <- resultats_m2()
    lasso_path <- res$lasso_path
    
    # Extraire les coefficients
    coefs <- as.matrix(lasso_path$beta)
    lambdas <- lasso_path$lambda
    feature_names <- rownames(coefs)
    
    # Créer un data frame long pour plotly
    p <- plot_ly()
    
    # Ajouter chaque feature comme une ligne
    colors <- rainbow(nrow(coefs), alpha = 0.7)
    for (i in 1:nrow(coefs)) {
      p <- p %>% add_trace(
        x = lambdas, y = coefs[i, ],
        type = "scatter", mode = "lines",
        line = list(color = colors[i], width = 1.5),
        name = feature_names[i],
        hoverinfo = "text",
        text = paste("Variable:", feature_names[i], "<br>Lambda:", round(lambdas, 4), "<br>Coef:", round(coefs[i, ], 2))
      )
    }
    
    # Ajouter ligne lambda optimal
    p <- p %>% add_trace(
      x = c(res$best_lambda, res$best_lambda), y = c(min(coefs), max(coefs)),
      type = "scatter", mode = "lines",
      line = list(color = "red", dash = "dash", width = 2),
      name = "Lambda optimal"
    )
    
    p %>% layout(
      title = "Chemin des Coefficients Lasso",
      xaxis = list(title = "Lambda (λ)", type = "log"),
      yaxis = list(title = "Coefficient"),
      showlegend = FALSE,
      hovermode = "closest"
    )
  })
  
  # --- GRAPHE 3 : PRÉDICTIONS ---
  output$m2_pred_plot <- renderPlotly({
    res <- resultats_m2()
    
    df_pred <- data.frame(
      Observed = res$y_test,
      Predicted = res$y_pred_test,
      Scenario = 1:length(res$y_test)
    )
    
    min_val <- min(c(df_pred$Observed, df_pred$Predicted))
    max_val <- max(c(df_pred$Observed, df_pred$Predicted))
    
    plot_ly() %>%
      add_trace(data = df_pred, x = ~Observed, y = ~Predicted,
                type = "scatter", mode = "markers",
                marker = list(size = 10, color = "#f39c12", opacity = 0.7),
                text = ~paste("Scénario:", Scenario,
                             "<br>Observé:", round(Observed, 0),
                             "<br>Prédit:", round(Predicted, 0)),
                hoverinfo = "text",
                name = "Prédictions") %>%
      add_trace(x = c(min_val, max_val), y = c(min_val, max_val),
                type = "scatter", mode = "lines",
                line = list(color = "red", dash = "dash"),
                name = "Parfait",
                hoverinfo = "none") %>%
      layout(title = paste("Prédictions Lasso - R² =", round(res$r2, 4)),
             xaxis = list(title = "Valeurs Observées (SMax)"),
             yaxis = list(title = "Prédictions Lasso"),
             showlegend = TRUE)
  })
  
  # --- GRAPHE 4 : RÉSIDUS ---
  output$m2_residuals <- renderPlotly({
    res <- resultats_m2()
    residus <- res$y_test - res$y_pred_test
    
    plot_ly(x = residus, type = "histogram", nbinsx = 20,
            marker = list(color = "#f39c12", line = list(color = "white", width = 1))) %>%
      layout(title = "Distribution des Résidus (Test)",
             xaxis = list(title = "Résidus (Observé - Prédit)"),
             yaxis = list(title = "Fréquence"),
             shapes = list(
               list(type = "line", x0 = 0, x1 = 0, y0 = 0, y1 = 1, yref = "paper",
                    line = list(color = "red", dash = "dash"))
             ))
  })
  
  # --- GRAPHE 5 : COEFFICIENTS SÉLECTIONNÉS ---
  output$m2_importance <- renderPlotly({
    res <- resultats_m2()
    coef_df <- res$coef_df
    
    if (nrow(coef_df) == 0) {
      return(plot_ly() %>% layout(title = "Aucune variable sélectionnée (tous les coefficients = 0)"))
    }
    
    p <- ggplot(coef_df, aes(x = reorder(feature, abs(coefficient)), y = coefficient,
                             fill = coefficient > 0,
                             text = paste("Variable:", feature,
                                          "<br>Coefficient:", round(coefficient, 2)))) +
      geom_bar(stat = "identity") +
      coord_flip() +
      scale_fill_manual(values = c("firebrick", "#27ae60"),
                        labels = c("Négatif", "Positif"),
                        name = "Effet") +
      labs(title = paste("Coefficients Lasso Sélectionnés (", nrow(coef_df), " variables)"),
           x = "Variable", y = "Coefficient (standardisé)") +
      theme_minimal() +
      theme(legend.position = "bottom")
    
    ggplotly(p, tooltip = "text") %>%
      layout(hoverlabel = list(bgcolor = "white"))
  })
  
  # --- MISE À JOUR SELECT INPUT 3D ---
  observe({
    res <- tryCatch(resultats_m2(), error = function(e) NULL)
    if (!is.null(res) && nrow(res$coef_df) > 0) {
      selected_vars <- as.character(res$coef_df$feature)
      updateSelectInput(session, "m2_var_3d_x", choices = selected_vars, selected = selected_vars[1])
      updateSelectInput(session, "m2_var_3d_y", choices = selected_vars, 
                        selected = if(length(selected_vars) > 1) selected_vars[2] else selected_vars[1])
      updateSelectInput(session, "m2_var_3d_color", choices = c("SMax", selected_vars), selected = "SMax")
    }
  })
  
  # --- GRAPHE 6 : 3D INTERACTIF ---
  output$m2_plot_3d <- renderPlotly({
    req(input$m2_var_3d_x, input$m2_var_3d_y, input$m2_var_3d_color)
    res <- resultats_m2()
    
    # Reconstruire les données originales
    X <- read.csv("data/extracted.csv")
    Y <- read.csv("data/data_Y.csv")
    if ("sc" %in% colnames(Y)) colnames(Y)[colnames(Y) == "sc"] <- "scenario_id"
    df <- inner_join(X, Y, by = "scenario_id")
    
    x_var <- input$m2_var_3d_x
    y_var <- input$m2_var_3d_y
    color_var <- input$m2_var_3d_color
    
    if (!(x_var %in% names(df)) || !(y_var %in% names(df))) {
      return(plot_ly() %>% layout(title = "Variables non disponibles"))
    }
    
    plot_ly(df, x = ~get(x_var), y = ~get(y_var), z = ~SMax,
            type = "scatter3d", mode = "markers",
            marker = list(
              size = 5,
              color = ~get(color_var),
              colorscale = "YlOrRd",
              showscale = TRUE,
              colorbar = list(title = color_var)
            ),
            text = ~paste(x_var, ":", round(get(x_var), 2),
                         "<br>", y_var, ":", round(get(y_var), 2),
                         "<br>SMax:", round(SMax, 0)),
            hoverinfo = "text") %>%
      layout(title = paste("Relations 3D :", x_var, "vs", y_var, "vs SMax"),
             scene = list(
               xaxis = list(title = x_var),
               yaxis = list(title = y_var),
               zaxis = list(title = "SMax")
             ))
  })
  
  # ========== MODÈLE 3 : RANDOM FOREST ==========
  
  resultats_m3 <- eventReactive(input$m3_run, {
    
    # ----------------------------
    # 1. CHARGEMENT & PRÉPARATION
    # ----------------------------
    X <- read.csv("data/extracted.csv")
    Y <- read.csv("data/data_Y.csv")
    
    # Jointure
    data_merged <- inner_join(X, Y, by = c("scenario_id" = "sc"))
    
    # Nettoyage : supprimer scenario_id, S_max, S_mean
    cols_to_remove <- c("scenario_id", "S_max", "S_mean")
    data_prep <- data_merged[, !(names(data_merged) %in% cols_to_remove)]
    
    # Suppression colonnes constantes
    col_stds <- sapply(data_prep, sd, na.rm = TRUE)
    data_clean <- data_prep[, col_stds > 0]
    
    # S'assurer que SMax est présent
    if (!"SMax" %in% names(data_clean)) {
      data_clean$SMax <- data_prep$SMax
    }
    
    # ----------------------------
    # 2. RANDOM FOREST INITIAL (toutes variables)
    # ----------------------------
    set.seed(123)
    ntree <- input$m3_ntree
    
    rf_initial <- randomForest(
      SMax ~ ., 
      data = data_clean, 
      ntree = ntree, 
      importance = TRUE
    )
    
    # Métriques initiales (OOB)
    r2_initial <- 1 - (sum((data_clean$SMax - rf_initial$predicted)^2) / 
                       sum((data_clean$SMax - mean(data_clean$SMax))^2))
    rmse_initial <- sqrt(mean((data_clean$SMax - rf_initial$predicted)^2))
    mae_initial <- mean(abs(data_clean$SMax - rf_initial$predicted))
    
    # ----------------------------
    # 3. IMPORTANCE DES VARIABLES
    # ----------------------------
    importance_df <- data.frame(
      Variable = rownames(importance(rf_initial)),
      IncMSE = importance(rf_initial)[, "%IncMSE"],
      IncNodePurity = importance(rf_initial)[, "IncNodePurity"]
    )
    importance_df <- importance_df[order(-importance_df$IncMSE), ]
    
    # ----------------------------
    # 4. FILTRAGE DES VARIABLES
    # ----------------------------
    seuil <- input$m3_seuil_importance
    vars_retenues <- importance_df$Variable[importance_df$IncMSE >= seuil]
    vars_eliminees <- importance_df$Variable[importance_df$IncMSE < seuil]
    
    # ----------------------------
    # 5. RANDOM FOREST OPTIMISÉ
    # ----------------------------
    if (length(vars_retenues) >= 2) {
      data_optimized <- data_clean[, c(as.character(vars_retenues), "SMax")]
      
      set.seed(123)
      rf_optimized <- randomForest(
        SMax ~ ., 
        data = data_optimized, 
        ntree = ntree, 
        importance = TRUE
      )
      
      # Métriques optimisées
      r2_optimized <- 1 - (sum((data_optimized$SMax - rf_optimized$predicted)^2) / 
                           sum((data_optimized$SMax - mean(data_optimized$SMax))^2))
      rmse_optimized <- sqrt(mean((data_optimized$SMax - rf_optimized$predicted)^2))
      mae_optimized <- mean(abs(data_optimized$SMax - rf_optimized$predicted))
      
      # Importance optimisée
      importance_opt <- data.frame(
        Variable = rownames(importance(rf_optimized)),
        IncMSE = importance(rf_optimized)[, "%IncMSE"]
      )
    } else {
      rf_optimized <- rf_initial
      r2_optimized <- r2_initial
      rmse_optimized <- rmse_initial
      mae_optimized <- mae_initial
      importance_opt <- importance_df
      data_optimized <- data_clean
    }
    
    list(
      rf_initial = rf_initial,
      rf_optimized = rf_optimized,
      r2_initial = r2_initial,
      rmse_initial = rmse_initial,
      mae_initial = mae_initial,
      r2_optimized = r2_optimized,
      rmse_optimized = rmse_optimized,
      mae_optimized = mae_optimized,
      importance_all = importance_df,
      importance_opt = importance_opt,
      vars_retenues = vars_retenues,
      vars_eliminees = vars_eliminees,
      data_clean = data_clean,
      data_optimized = data_optimized,
      observed = data_clean$SMax,
      predicted_initial = rf_initial$predicted,
      predicted_optimized = rf_optimized$predicted,
      ntree = ntree,
      seuil = seuil,
      # Pour comparaison globale
      r2 = r2_optimized,
      rmse = rmse_optimized,
      mae = mae_optimized
    )
  })
  
  # --- MÉTRIQUES INITIALES ---
  output$m3_metrics_initial <- renderTable({
    res <- resultats_m3()
    data.frame(
      Métrique = c("R² OOB", "RMSE", "MAE"),
      Valeur = c(
        paste0(round(res$r2_initial * 100, 2), "%"),
        round(res$rmse_initial, 0),
        round(res$mae_initial, 0)
      )
    )
  })
  
  # --- MÉTRIQUES OPTIMISÉES ---
  output$m3_metrics_optimized <- renderTable({
    res <- resultats_m3()
    data.frame(
      Métrique = c("R² OOB", "RMSE", "MAE"),
      Valeur = c(
        paste0(round(res$r2_optimized * 100, 2), "%"),
        round(res$rmse_optimized, 0),
        round(res$mae_optimized, 0)
      )
    )
  })
  
  # --- INFO VARIABLES ---
  output$m3_vars_info <- renderUI({
    res <- resultats_m3()
    HTML(paste0(
      "<p><b>Variables retenues (", length(res$vars_retenues), ") :</b> ",
      paste(res$vars_retenues[1:min(10, length(res$vars_retenues))], collapse = ", "),
      if(length(res$vars_retenues) > 10) "..." else "",
      "</p>",
      "<p><b>Variables éliminées (", length(res$vars_eliminees), ") :</b> considérées comme bruit (%IncMSE < ", res$seuil, "%)</p>"
    ))
  })
  
  # --- GRAPHE 1 : CONVERGENCE ---
  output$m3_convergence <- renderPlotly({
    res <- resultats_m3()
    rf <- res$rf_initial
    
    # Extraire l'erreur MSE par nombre d'arbres
    mse_values <- rf$mse
    n_trees <- 1:length(mse_values)
    
    plot_ly(x = n_trees, y = mse_values, type = "scatter", mode = "lines",
            line = list(color = "#e74c3c", width = 2),
            name = "Erreur MSE") %>%
      add_trace(x = c(150, 150), y = c(min(mse_values), max(mse_values)),
                type = "scatter", mode = "lines",
                line = list(color = "green", dash = "dash"),
                name = "Stabilisation (~150)") %>%
      add_trace(x = c(res$ntree, res$ntree), y = c(min(mse_values), max(mse_values)),
                type = "scatter", mode = "lines",
                line = list(color = "blue", dash = "dot"),
                name = paste("Choisi (", res$ntree, ")")) %>%
      layout(title = "Convergence de l'Erreur OOB",
             xaxis = list(title = "Nombre d'Arbres"),
             yaxis = list(title = "Erreur MSE (OOB)"),
             showlegend = TRUE)
  })
  
  # --- GRAPHE 2 : IMPORTANCE DES VARIABLES ---
  output$m3_importance <- renderPlotly({
    res <- resultats_m3()
    imp <- res$importance_all
    
    # Marquer les variables retenues
    imp$Statut <- ifelse(imp$Variable %in% res$vars_retenues, "Retenue", "Éliminée")
    
    p <- ggplot(imp, aes(x = reorder(Variable, IncMSE), y = IncMSE, fill = Statut,
                         text = paste("Variable:", Variable,
                                      "<br>%IncMSE:", round(IncMSE, 2),
                                      "<br>Statut:", Statut))) +
      geom_bar(stat = "identity") +
      geom_hline(yintercept = res$seuil, linetype = "dashed", color = "red", size = 1) +
      coord_flip() +
      scale_fill_manual(values = c("Retenue" = "#27ae60", "Éliminée" = "#bdc3c7")) +
      labs(title = paste("Importance des Variables - Seuil =", res$seuil, "%"),
           x = "Variable", y = "%IncMSE") +
      theme_minimal() +
      theme(axis.text.y = element_text(size = 7))
    
    ggplotly(p, tooltip = "text") %>%
      layout(hoverlabel = list(bgcolor = "white"))
  })
  
  # --- GRAPHE 3 : PRÉDICTIONS OOB ---
  output$m3_pred_plot <- renderPlotly({
    res <- resultats_m3()
    
    df_pred <- data.frame(
      Observed = res$observed,
      Predicted = res$predicted_optimized,
      Scenario = 1:length(res$observed)
    )
    
    # Calculer la ligne parfaite
    min_val <- min(c(df_pred$Observed, df_pred$Predicted))
    max_val <- max(c(df_pred$Observed, df_pred$Predicted))
    
    plot_ly() %>%
      add_trace(data = df_pred, x = ~Observed, y = ~Predicted, 
                type = "scatter", mode = "markers",
                marker = list(size = 8, color = "#3498db", opacity = 0.7),
                text = ~paste("Scénario:", Scenario,
                             "<br>Observé:", round(Observed, 0),
                             "<br>Prédit:", round(Predicted, 0)),
                hoverinfo = "text",
                name = "Prédictions") %>%
      add_trace(x = c(min_val, max_val), y = c(min_val, max_val),
                type = "scatter", mode = "lines",
                line = list(color = "red", dash = "dash"),
                name = "Parfait",
                hoverinfo = "none") %>%
      layout(title = paste("Prédictions OOB - R² =", round(res$r2_optimized * 100, 2), "%"),
             xaxis = list(title = "Valeurs Observées (SMax)"),
             yaxis = list(title = "Prédictions OOB"),
             showlegend = TRUE)
  })
  
  
  # --- GRAPHE 6 : PIE IMPORTANCE ---
  output$m3_importance_pie <- renderPlotly({
    res <- resultats_m3()
    imp <- res$importance_all
    
    # Top 10 variables
    top_vars <- head(imp, 10)
    autres <- sum(imp$IncMSE[11:nrow(imp)])
    
    if (nrow(imp) > 10) {
      pie_data <- data.frame(
        Variable = c(as.character(top_vars$Variable), "Autres"),
        Importance = c(top_vars$IncMSE, autres)
      )
    } else {
      pie_data <- data.frame(
        Variable = as.character(top_vars$Variable),
        Importance = top_vars$IncMSE
      )

    }
    
    plot_ly(pie_data, labels = ~Variable, values = ~Importance, type = "pie",
            textinfo = "label+percent",
            marker = list(colors = RColorBrewer::brewer.pal(min(11, nrow(pie_data)), "Set3"))) %>%
      layout(title = "Top 10 Variables les Plus Importantes")
  })
  observe({
    res <- tryCatch(resultats_m3(), error = function(e) NULL)
    if (!is.null(res)) {
      top_vars <- as.character(head(res$importance_all$Variable, 15))
      updateSelectInput(session, "m3_var_3d_x", choices = top_vars, selected = top_vars[1])
      updateSelectInput(session, "m3_var_3d_y", choices = top_vars, selected = top_vars[2])
      updateSelectInput(session, "m3_var_3d_color", choices = c("SMax", top_vars), selected = "SMax")
    }
  })
  
 
  output$m3_plot_3d <- renderPlotly({
    req(input$m3_var_3d_x, input$m3_var_3d_y, input$m3_var_3d_color)
    res <- resultats_m3()
    df <- res$data_clean
    
    x_var <- input$m3_var_3d_x
    y_var <- input$m3_var_3d_y
    color_var <- input$m3_var_3d_color
    
    plot_ly(df, x = ~get(x_var), y = ~get(y_var), z = ~SMax,
            type = "scatter3d", mode = "markers",
            marker = list(
              size = 5,
              color = ~get(color_var),
              colorscale = "Viridis",
              showscale = TRUE,
              colorbar = list(title = color_var)
            ),
            text = ~paste(x_var, ":", round(get(x_var), 2),
                         "<br>", y_var, ":", round(get(y_var), 2),
                         "<br>SMax:", round(SMax, 0)),
            hoverinfo = "text") %>%
      layout(title = paste("Relations 3D :", x_var, "vs", y_var, "vs SMax"),
             scene = list(
               xaxis = list(title = x_var),
               yaxis = list(title = y_var),
               zaxis = list(title = "SMax")
             ))
  })
  
  # ========== COMPARAISON DES 3 MODÈLES ==========
  
  output$plot_comparison_r2 <- renderPlotly({
    # Récupérer les résultats (avec valeurs par défaut si pas encore lancés)
    r2_m1 <- tryCatch(results_m1()$r2, error = function(e) 0)
    r2_m2 <- tryCatch(resultats_m2()$r2, error = function(e) 0)
    r2_m3 <- tryCatch(resultats_m3()$r2, error = function(e) 0)
    
    df_comp <- data.frame(
      Méthode = c("Méthode 1", "Méthode 2", "Méthode 3"),
      R2 = c(r2_m1, r2_m2, r2_m3)
    )
    
    plot_ly(df_comp, x = ~Méthode, y = ~R2, type = "bar",
            marker = list(color = c("#27ae60", "#f39c12", "#e74c3c"))) %>%
      layout(title = "Comparaison R² (plus élevé = meilleur)",
             yaxis = list(title = "R²", range = c(0, 1)))
  })
  
  output$plot_comparison_rmse <- renderPlotly({
    rmse_m1 <- tryCatch(results_m1()$rmse, error = function(e) 0)
    rmse_m2 <- tryCatch(resultats_m2()$rmse, error = function(e) 0)
    rmse_m3 <- tryCatch(resultats_m3()$rmse, error = function(e) 0)
    
    df_comp <- data.frame(
      Méthode = c("Méthode 1", "Méthode 2", "Méthode 3"),
      RMSE = c(rmse_m1, rmse_m2, rmse_m3)
    )
    
    plot_ly(df_comp, x = ~Méthode, y = ~RMSE, type = "bar",
            marker = list(color = c("#27ae60", "#f39c12", "#e74c3c"))) %>%
      layout(title = "Comparaison RMSE (plus faible = meilleur)",
             yaxis = list(title = "RMSE"))
  })
  
  output$table_comparison <- renderDT({
    r2_m1 <- tryCatch(results_m1()$r2, error = function(e) NA)
    r2_m2 <- tryCatch(resultats_m2()$r2, error = function(e) NA)
    r2_m3 <- tryCatch(resultats_m3()$r2, error = function(e) NA)
    
    rmse_m1 <- tryCatch(results_m1()$rmse, error = function(e) NA)
    rmse_m2 <- tryCatch(resultats_m2()$rmse, error = function(e) NA)
    rmse_m3 <- tryCatch(resultats_m3()$rmse, error = function(e) NA)
    
    mae_m1 <- tryCatch(results_m1()$mae, error = function(e) NA)
    mae_m2 <- tryCatch(resultats_m2()$mae, error = function(e) NA)
    mae_m3 <- tryCatch(resultats_m3()$mae, error = function(e) NA)
    
    df_comp <- data.frame(
      Méthode = c("Méthode 1", "Méthode 2", "Méthode 3"),
      R2 = round(c(r2_m1, r2_m2, r2_m3), 3),
      RMSE = round(c(rmse_m1, rmse_m2, rmse_m3), 2),
      MAE = round(c(mae_m1, mae_m2, mae_m3), 2),
      check.names = FALSE
    )
    
    datatable(df_comp, options = list(dom = 't')) %>%
      formatStyle('R²', backgroundColor = styleInterval(c(0.8, 0.9), c('white', '#d4edda', '#28a745'))) %>%
      formatStyle('RMSE', backgroundColor = styleInterval(c(10000, 15000), c('#28a745', '#d4edda', 'white')))
  })
  
  output$best_method <- renderText({
    r2_m1 <- tryCatch(results_m1()$r2, error = function(e) 0)
    r2_m2 <- tryCatch(resultats_m2()$r2, error = function(e) 0)
    r2_m3 <- tryCatch(resultats_m3()$r2, error = function(e) 0)
    
    r2_vals <- c(r2_m1, r2_m2, r2_m3)
    best <- which.max(r2_vals)
    
    paste("Méthode", best, "(R² =", round(r2_vals[best], 3), ")")
  })
  
  output$key_points <- renderUI({
    HTML("
      <ul>
        <li>Lancez les 3 modèles pour voir la comparaison</li>
        <li>Le meilleur modèle est celui avec le R² le plus élevé</li>
        <li>Vérifiez aussi le RMSE (plus faible = meilleur)</li>
      </ul>
    ")
  })
  
  output$plot_comparison_predictions <- renderPlotly({
    # Graphique combiné des prédictions (si Modèle 1 est lancé)
    res <- tryCatch(results_m1(), error = function(e) NULL)
    if(!is.null(res)) {
      plot_ly(x = res$obs_test, y = res$pred_test,
              type = "scatter", mode = "markers",
              name = "Modèle 1",
              marker = list(size = 8, opacity = 0.6)) %>%
        add_trace(x = range(res$obs_test),
                  y = range(res$obs_test),
                  type = "scatter", mode = "lines",
                  line = list(color = "red", dash = "dash"),
                  name = "Parfait") %>%
        layout(title = "Comparaison visuelle des prédictions",
               xaxis = list(title = "Valeurs réelles"),
               yaxis = list(title = "Valeurs prédites"))
    } else {
      plot_ly() %>%
        layout(title = "Lancez au moins un modèle pour voir les prédictions")
    }
  })
  
}

shinyApp(ui = ui, server = server)