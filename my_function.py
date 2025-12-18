import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib_venn import venn2

# Vos imports personnalisés existants
from src.data.load_data import *
from src.utils.results_utils import *


def plot_most_wanted_plotly(
    df: pd.DataFrame,
    source_col="SOURCE_SUBREDDIT",
    sentiment_col="LINK_SENTIMENT",
    top_n=15
):
    """
    Génère un graphique interactif 'Diverging Bar Chart' avec Plotly.
    Identifie les agresseurs (Gauche/Rouge) vs les diplomates (Droite/Bleu).
    """
    
    # --- 1. Préparation des données (Identique) ---
    stats = df.groupby([source_col, sentiment_col]).size().unstack(fill_value=0)
    
    if -1 not in stats.columns: stats[-1] = 0
    if 1 not in stats.columns: stats[1] = 0
    
    stats = stats.rename(columns={-1: 'negative_count', 1: 'positive_count'})
    stats['total_count'] = stats['negative_count'] + stats['positive_count']
    stats['toxicity_ratio'] = stats['negative_count'] / stats['total_count']
    
    # Trier par nombre d'attaques (les pires en haut)
    top_aggressors = stats.sort_values('negative_count', ascending=True).tail(top_n)
    
    # --- 2. Construction du Plot ---
    fig = go.Figure()

    # -- Trace 1 : Attaques (Négatif, vers la gauche) --
    # On passe les valeurs en négatif pour qu'elles partent à gauche
    fig.add_trace(go.Bar(
        y=top_aggressors.index,
        x=-top_aggressors['negative_count'], 
        orientation='h',
        name='Attacks (Negative)',
        marker=dict(color='#d62728', line=dict(width=0)), # Rouge
        # Customdata pour le hover: [Vraie valeur positive, Ratio %]
        customdata=np.stack((top_aggressors['negative_count'], top_aggressors['toxicity_ratio']*100), axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Attacks: %{customdata[0]:,}<br>" +
            "Toxic Ratio: %{customdata[1]:.1f}%<br>" +
            "<extra></extra>"
        )
    ))

    # -- Trace 2 : Diplomatie (Positif, vers la droite) --
    fig.add_trace(go.Bar(
        y=top_aggressors.index,
        x=top_aggressors['positive_count'],
        orientation='h',
        name='Diplomacy (Positive)',
        marker=dict(color='#1f77b4', line=dict(width=0)), # Bleu
        customdata=np.stack((top_aggressors['positive_count'], top_aggressors['toxicity_ratio']*100), axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Diplomacy: %{customdata[0]:,}<br>" +
            "Toxic Ratio: %{customdata[1]:.1f}%<br>" +
            "<extra></extra>"
        )
    ))

    # --- 3. Annotations (Le % de Toxicité à droite) ---
    annotations = []
    
    # Calculer la limite max pour placer le texte un peu après la barre la plus longue
    max_val = max(top_aggressors['positive_count'].max(), top_aggressors['negative_count'].max())
    
    for subreddit, row in top_aggressors.iterrows():
        # Annotation du pourcentage
        annotations.append(dict(
            x=row['positive_count'], 
            y=subreddit,
            text=f" <b>{row['toxicity_ratio']*100:.1f}% Toxic</b>",
            xanchor='left',
            yanchor='middle',
            showarrow=False,
            font=dict(color='#d62728', size=12) # Texte en rouge pour rappeler le danger
        ))
        
        # (Optionnel) Annotation du nombre d'attaques à gauche (dans la barre rouge)
        if row['negative_count'] > max_val * 0.1: # Seulement si la barre est assez grande
            annotations.append(dict(
                x=-row['negative_count'] / 2,
                y=subreddit,
                text=str(int(row['negative_count'])),
                xanchor='center',
                yanchor='middle',
                showarrow=False,
                font=dict(color='white', size=10)
            ))

    # --- 4. Layout "Pro" ---
    
    # Astuce pour l'axe X : faire en sorte que -5000 s'affiche "5000"
    # On génère des ticks dynamiques
    tick_vals = np.linspace(-max_val, max_val, 7)
    tick_text = [f"{int(abs(x))}" for x in tick_vals]

    fig.update_layout(
        title=dict(
            text=f"<b>The 'Most Wanted' List:</b> Toxicity vs. Diplomacy (Top {top_n})",
            font=dict(size=20)
        ),
        barmode='relative', # Permet d'aligner sur le 0 central
        xaxis=dict(
            title="Number of Hyperlinks",
            tickvals=tick_vals,
            ticktext=tick_text,
            gridcolor='rgba(0,0,0,0.1)' # Grille légère
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12, family="Arial Black") # Police grasse pour les noms
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=annotations,
        template="plotly_white",
        height=600 + (top_n * 20), # Hauteur adaptative
        margin=dict(r=100) # Marge à droite pour le texte des %
    )
    
    # Ligne verticale centrale
    fig.add_vline(x=0, line_width=1, line_color="black")

    fig.show()





def plot_tactical_map_interactive(merge_df, min_interactions=10):
    """
    Génère une 'Tactical Map' interactive avec Plotly à partir du DataFrame fusionné.
    Corrigé pour la palette de couleurs Plotly.
    """
    print("Construction de la Tactical Map Interactive...")
    
    # --- 1. Filtrage des données ---
    neg_df = merge_df[merge_df['LINK_SENTIMENT'] == -1]
    
    out_degree = neg_df['SOURCE_SUBREDDIT'].value_counts().rename('out_degree')
    in_degree = neg_df['TARGET_SUBREDDIT'].value_counts().rename('in_degree')
    
    stats = pd.DataFrame(out_degree).join(in_degree, how='outer').fillna(0)
    stats['total_activity'] = stats['out_degree'] + stats['in_degree']
    
    # Filtrer le bruit
    stats = stats[stats['total_activity'] >= min_interactions].copy()
    
    # Calculer le Sniper Ratio
    stats['sniper_ratio'] = stats['out_degree'] / stats['total_activity']
    stats['subreddit_name'] = stats.index

    # --- 2. Astuce Log-Scale ---
    stats['plot_out'] = stats['out_degree'] + 0.5
    stats['plot_in'] = stats['in_degree'] + 0.5

    # --- 3. Création du Plot ---
    fig = px.scatter(
        stats,
        x="plot_out", 
        y="plot_in",
        color="sniper_ratio",
        hover_name="subreddit_name",
        custom_data=["out_degree", "in_degree", "sniper_ratio"],
        log_x=True, 
        log_y=True,
        # CORRECTION ICI : 'RdBu_r' (Red-Blue reversed) donne Bleu -> Rouge
        color_continuous_scale="RdBu_r", 
        title="The Tactical Map: Snipers, Victims, and War Zones",
        labels={"sniper_ratio": "Sniper Ratio"},
        template="plotly_white"
    )

    # --- 4. Customisation du Tooltip (Hover) ---
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>" +
            "Attacks Launched: %{customdata[0]:.0f}<br>" + 
            "Attacks Received: %{customdata[1]:.0f}<br>" + 
            "Sniper Ratio: %{customdata[2]:.2f}"
        ),
        marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey'), opacity=0.8)
    )

    # --- 5. Annotations des Zones ---
    x_max = stats['plot_out'].max()
    y_max = stats['plot_in'].max()
    min_val = 0.5 

    # Zone Snipers (Bas-Droite)
    fig.add_annotation(
        x=np.log10(x_max), y=np.log10(min_val), 
        text="<b>ZONE: SNIPERS</b><br>(Bullies)",
        showarrow=False, font=dict(color="darkred", size=14),
        bgcolor="rgba(255,255,255,0.8)", xanchor="right", yanchor="bottom"
    )
    
    # Zone Punching Bags (Haut-Gauche)
    fig.add_annotation(
        x=np.log10(min_val), y=np.log10(y_max),
        text="<b>ZONE: VICTIMS</b><br>(Punching Bags)",
        showarrow=False, font=dict(color="darkblue", size=14),
        bgcolor="rgba(255,255,255,0.8)", xanchor="left", yanchor="top"
    )
    
    # Zone War Zones (Haut-Droite)
    fig.add_annotation(
        x=np.log10(x_max), y=np.log10(y_max),
        text="<b>ZONE: WAR</b><br>(Total Chaos)",
        showarrow=False, font=dict(color="purple", size=14),
        bgcolor="rgba(255,255,255,0.8)", xanchor="right", yanchor="top"
    )

    # --- 6. Layout Final ---
    fig.update_layout(
        height=750,
        xaxis_title="Aggression Score (Attaques Lancées)",
        yaxis_title="Victim Score (Attaques Reçues)",
        coloraxis_colorbar=dict(title="Sniper Ratio<br>(0=Victim, 1=Sniper)"),
        hovermode="closest"
    )

    fig.show()





def plot_constancy_rainplot(merge_df, min_attacks=50, top_n=40, opportunist_share=0.7):
    """
    Génère le Dot Plot.
    - opportunist_share=0.7 signifie que 70% du graphique sera dédié aux 'Opportunistes' (Toxicité aiguë).
    """
    print(f"Génération avec {int(opportunist_share*100)}% d'Opportunistes...")

    # 1. Préparation (Identique)
    neg_df = merge_df[merge_df['LINK_SENTIMENT'] == -1].copy()
    if not pd.api.types.is_datetime64_any_dtype(neg_df['TIMESTAMP']):
        neg_df['TIMESTAMP'] = pd.to_datetime(neg_df['TIMESTAMP'])
    
    monthly = neg_df.groupby([
        'SOURCE_SUBREDDIT', 
        pd.Grouper(key='TIMESTAMP', freq='MS')
    ]).size().reset_index(name='attacks')

    # 2. Calcul des Métriques
    stats = monthly.groupby('SOURCE_SUBREDDIT').agg(
        total_attacks=('attacks', 'sum'),
        active_months=('attacks', 'count'), 
        max_burst=('attacks', 'max')       
    )
    stats = stats[stats['total_attacks'] >= min_attacks]
    
    # --- 3. SÉLECTION ASYMÉTRIQUE (La Modification) ---
    
    # Calcul du nombre de slots pour chaque catégorie
    n_opportunists = int(top_n * opportunist_share) # Ex: 40 * 0.7 = 28
    n_serial = top_n - n_opportunists               # Ex: 40 - 28 = 12
    
    # Groupe A : Les Serial Killers (Les plus constants)
    serial_killers = stats.sort_values('active_months', ascending=False).head(n_serial).index.tolist()
    
    # Groupe B : Les Opportunistes (Les plus gros pics parmi ceux qui restent)
    remaining = stats[~stats.index.isin(serial_killers)]
    opportunists = remaining.sort_values('max_burst', ascending=False).head(n_opportunists).index.tolist()
    
    # Fusion
    final_selection = serial_killers + opportunists
    
    # Tri visuel (du moins constant au plus constant)
    sorted_selection = stats.loc[final_selection].sort_values('active_months', ascending=True).index.tolist()
    plot_df = monthly[monthly['SOURCE_SUBREDDIT'].isin(sorted_selection)].copy()

    # 4. Plotting
    fig = px.scatter(
        plot_df,
        x="TIMESTAMP",
        y="SOURCE_SUBREDDIT",
        size="attacks",       
        color="attacks",      
        hover_name="SOURCE_SUBREDDIT",
        size_max=25, 
        
        # Échelle de couleur Saumon -> Rouge Sang
        color_continuous_scale=[
            [0.0, "salmon"], 
            [0.5, "red"], 
            [1.0, "darkred"]
        ],
        
        title=f"Conflict Spectrum: Focusing on Acute Toxicity (Opportunists)",
        template="plotly_white",
        category_orders={"SOURCE_SUBREDDIT": sorted_selection} 
    )

    # Styling
    fig.update_traces(
        marker=dict(
            sizemin=5,
            line=dict(width=1, color='DarkSlateGrey'), 
            opacity=0.9
        )
    )

    fig.update_layout(
        height=int(20 * top_n) + 100, # Hauteur adaptative selon le nombre de lignes
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickfont=dict(size=10)),
        xaxis=dict(showgrid=False),
        coloraxis_showscale=False, 
        margin=dict(l=150)
    )
    
    # Annotations
    fig.add_annotation(
        x=plot_df['TIMESTAMP'].max(), y=sorted_selection[-1],
        text=f"<b>SERIAL KILLERS</b><br>(Top {n_serial})",
        showarrow=False, xanchor="left", xshift=10, font=dict(color="darkred", size=10)
    )
    
    fig.add_annotation(
        x=plot_df['TIMESTAMP'].max(), y=sorted_selection[0],
        text=f"<b>OPPORTUNISTS</b><br>(Top {n_opportunists})",
        showarrow=False, xanchor="left", xshift=10, font=dict(color="orangered", size=10)
    )

    fig.show()


def plot_victim_semesters_slider(merge_df, top_n=10):
    """
    Version STABILISÉE : Empêche le 'saut' du graphique (Wobble effect).
    - Marge Gauche FIXE (pour les longs noms de subreddits).
    - Axe X FIXE (autorange desactive).
    - Transition fluide.
    """
    print("Génération du Plot Interactif Stabilisé...")

    # 1. Préparation (identique)
    neg_df = merge_df[merge_df['LINK_SENTIMENT'] == -1].copy()
    if not pd.api.types.is_datetime64_any_dtype(neg_df['TIMESTAMP']):
        neg_df['TIMESTAMP'] = pd.to_datetime(neg_df['TIMESTAMP'], utc=True).dt.tz_localize(None)
    
    neg_df = neg_df[neg_df['TIMESTAMP'].dt.year >= 2014]
    neg_df['Year'] = neg_df['TIMESTAMP'].dt.year
    neg_df['Semester'] = np.where(neg_df['TIMESTAMP'].dt.month <= 6, 'H1', 'H2')
    neg_df['Period'] = neg_df['Year'].astype(str) + " - " + neg_df['Semester']
    
    periods = sorted(neg_df['Period'].unique())

    # 2. Max Global
    global_counts = neg_df.groupby(['Period', 'TARGET_SUBREDDIT']).size()
    if global_counts.empty:
        print("Aucune donnée trouvée.")
        return
    global_max = global_counts.max()

    # 3. Figure Initiale
    initial_period = periods[0]
    initial_data = neg_df[neg_df['Period'] == initial_period]['TARGET_SUBREDDIT'].value_counts().head(top_n).sort_values(ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=initial_data.values,
        y=initial_data.index,
        orientation='h',
        text=initial_data.values,
        textposition='outside',
        marker=dict(
            color=initial_data.values,
            colorscale='Reds',
            cmin=0,
            cmax=global_max
        ),
        name=initial_period
    ))

    # 4. Frames (avec redessin forcé pour éviter les glitchs)
    frames = []
    for period in periods:
        subset = neg_df[neg_df['Period'] == period]
        counts = subset['TARGET_SUBREDDIT'].value_counts().head(top_n).sort_values(ascending=True)
        
        frames.append(go.Frame(
            data=[go.Bar(
                x=counts.values,
                y=counts.index,
                text=counts.values,
                marker=dict(color=counts.values) 
            )],
            name=period,
            layout=go.Layout(title_text=f"<b>Top Victims: {period}</b>")
        ))

    fig.frames = frames

    # 5. Layout STABILISÉ (C'est ici que la magie opère)
    fig.update_layout(
        title=f"<b>The Escalation: {initial_period}</b>",
        template="plotly_white",
        height=600,
        
        # --- FIX 1 : Marge Gauche Fixe ---
        # On réserve 200px pour les noms, comme ça le graphique ne 'danse' pas
        # si un nom est court puis long.
        margin=dict(l=200, r=50, t=100, b=50),
        
        # --- FIX 2 : Axe X Verrouillé ---
        xaxis=dict(
            range=[0, global_max * 1.15], 
            title="Number of Attacks Received",
            autorange=False,  # INTERDIT à Plotly de changer l'échelle
            fixedrange=True   # Empêche le zoom utilisateur qui casse tout
        ),
        yaxis=dict(title=""), # On enlève le titre Y pour gagner de la place
        
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Period: ",
                "visible": True,
                "xanchor": "right"
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f.name],
                        {
                            "frame": {"duration": 0, "redraw": True}, # Redraw True est vital ici
                            "mode": "immediate",
                            "transition": {"duration": 0} # Pas de transition floue
                        }
                    ],
                    "label": f.name,
                    "method": "animate"
                }
                for f in frames
            ]
        }]
    )

    fig.show()



def visualiser_ban_multi_plots(df, subreddit_list):
    """
    Génère une figure unique contenant plusieurs sous-graphiques avec mention 'BAN'.
    Corrige le décalage temporel en détectant la fin du VOLUME réel.
    """
    
    # 1. Gestion de la date de fin globale
    if not pd.api.types.is_datetime64_any_dtype(df['TIMESTAMP']):
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    end_date = df['TIMESTAMP'].max()
    
    if isinstance(subreddit_list, str):
        subreddit_list = [subreddit_list]

    # Filtrage préalable
    valid_subreddits = []
    for sub in subreddit_list:
        if not df[df['SOURCE_SUBREDDIT'] == sub].empty:
            valid_subreddits.append(sub)
        else:
            print(f"⚠️  Le subreddit '{sub}' est introuvable. Ignoré.")
    
    n_subs = len(valid_subreddits)
    if n_subs == 0:
        print("Aucun subreddit valide à afficher.")
        return

    # 2. Création de la Figure
    fig, axes = plt.subplots(nrows=n_subs, ncols=1, figsize=(14, 4 * n_subs), sharex=True)

    if n_subs == 1:
        axes = [axes]

    print(f"--- Génération de la planche pour {n_subs} subreddits ---\n")

    # 3. Boucle sur les axes
    for ax, subreddit_name in zip(axes, valid_subreddits):
        
        # A. Préparation
        sub_data = df[df['SOURCE_SUBREDDIT'] == subreddit_name].copy()
        
        if not pd.api.types.is_datetime64_any_dtype(sub_data['TIMESTAMP']):
            sub_data['TIMESTAMP'] = pd.to_datetime(sub_data['TIMESTAMP'])
        sub_data = sub_data.set_index('TIMESTAMP')

        # B. Catégorisation
        sub_data['is_pos_post'] = (sub_data['Positive sentiment calculated by VADER'] >= sub_data['Negative sentiment calculated by VADER']).astype(int)
        sub_data['is_neg_post'] = (sub_data['Negative sentiment calculated by VADER'] > sub_data['Positive sentiment calculated by VADER']).astype(int)

        # C. Resampling
        weekly_stats = sub_data.resample('W').agg({'is_pos_post': 'sum', 'is_neg_post': 'sum'})

        # --- CORRECTION DU DÉCALAGE ---
        # Au lieu de prendre le dernier timestamp brut, on cherche la dernière semaine ACTIVE.
        # On considère qu'une semaine est active si elle a au moins 1 post.
        weeks_with_activity = weekly_stats[ (weekly_stats['is_pos_post'] > 0) | (weekly_stats['is_neg_post'] > 0) ]
        
        if not weeks_with_activity.empty:
            # On prend la fin de la dernière semaine active
            last_real_activity = weeks_with_activity.index.max()
        else:
            # Fallback (filet de sécurité)
            last_real_activity = sub_data.index.max()

        # D. Extension Temporelle
        full_range = pd.date_range(start=weekly_stats.index.min(), end=end_date, freq='W')
        weekly_stats = weekly_stats.reindex(full_range)
        weekly_stats = weekly_stats.fillna(0)

        # E. Plotting
        ax.stackplot(weekly_stats.index,
                      weekly_stats['is_pos_post'],
                      weekly_stats['is_neg_post'],
                      labels=['Positive', 'Negative'],
                      colors=['#1f77b4', '#d62728'], alpha=0.8)
        
        # --- F. ZONE DE BAN & TEXTE ---
        
        # 1. Dessiner la zone grise (à partir de la semaine corrigée)
        ax.axvspan(last_real_activity, end_date, color='grey', alpha=0.2)
        
        # 2. Calculer le milieu de la zone temporelle pour centrer le texte
        duration_ban = end_date - last_real_activity
        mid_point_date = last_real_activity + duration_ban / 2
        
        # 3. Ajouter le texte "BAN"
        ax.text(mid_point_date, 0.5, "BAN", 
                color='red', fontsize=22, fontweight='bold', 
                ha='center', va='center', rotation=0,
                transform=ax.get_xaxis_transform())

        # Esthétique
        ax.set_title(f"r/{subreddit_name} (End of activity : {last_real_activity.date()})", fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel("Vol. Posts")
        ax.grid(True, alpha=0.3)
        
        if ax == axes[0]:
            ax.legend(loc='upper left')

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

def detecter_subreddits_bannis(
    df: pd.DataFrame, 
    time_col='TIMESTAMP', 
    sub_col='SOURCE_SUBREDDIT', 
    days_silence_threshold=300, 
    min_posts_threshold=100
):
    """
    Identifie les subreddits qui ont soudainement cessé d'être actifs avant la fin de la collecte
    (potentiellement bannis ou devenus privés).
    
    Returns:
        pd.DataFrame: DataFrame contenant uniquement les subreddits suspects + métadonnées.
    """
    # 1. Travail sur une copie pour ne pas modifier le DF original
    df_work = df.copy()
    
    # Conversion sécurisée en datetime
    if not pd.api.types.is_datetime64_any_dtype(df_work[time_col]):
        df_work[time_col] = pd.to_datetime(df_work[time_col])

    # 2. Déterminer la fin de la collecte globale
    global_max_date = df_work[time_col].max()
    cutoff_date = global_max_date - pd.Timedelta(days=days_silence_threshold)

    # 3. Agrégation par Subreddit
    # On calcule : premier post, dernier post, et nombre total de posts
    lifecycle = df_work.groupby(sub_col).agg(
        First_Seen=(time_col, 'min'),
        Last_Seen=(time_col, 'max'),
        Total_Activity=(time_col, 'count') # Compte le nombre de lignes
    ).reset_index()

    # 4. Filtrage : "Silence prématuré" ET "Gros volume passé"
    mask_silence = lifecycle['Last_Seen'] < cutoff_date
    mask_activity = lifecycle['Total_Activity'] > min_posts_threshold
    
    banned_candidates = lifecycle[mask_silence & mask_activity].copy()

    # 5. Info supplémentaire : Durée de vie en jours
    banned_candidates['Lifespan_Days'] = (
        banned_candidates['Last_Seen'] - banned_candidates['First_Seen']
    ).dt.days

    # Tri par date de disparition (les plus récents d'abord)
    return banned_candidates.sort_values(by='Last_Seen', ascending=False)



def visualiser_sarcasme_only_red(df, sample_size=5000):
    """
    Génère un Scatter Plot interactif montrant UNIQUEMENT les liens à intention négative (Rouge).
    Met en évidence :
    1. Le Sarcasme (Faux Positifs VADER)
    2. L'Honnêteté (Vrais Négatifs VADER - Attaques directes)
    """
    
    # 1. Filtrage STRICT : Uniquement les liens négatifs (-1)
    plot_df = df[df['LINK_SENTIMENT'] == -1].copy()
    
    if len(plot_df) == 0:
        print("⚠️ Aucune donnée avec LINK_SENTIMENT == -1 trouvée.")
        return

    # Sampling
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=42)
        print(f"ℹ️ Affichage d'un échantillon aléatoire de {sample_size} liens négatifs.")

    plot_df['Link Context'] = 'Negative Link (Attack/Drama)'

    # 2. Construction du Graphique
    fig = px.scatter(
        plot_df, 
        x="Compound sentiment calculated by VADER", 
        y="Positive sentiment calculated by VADER", 
        color="Link Context", 
        hover_data=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'], 
        title="<b>Sarcasm vs. Honesty:</b> How VADER interprets Hostile Intent",
        color_discrete_map={'Negative Link (Attack/Drama)': '#d62728'},
        opacity=0.6,
        template="plotly_white",
        height=700
    )

    # Calcul de la hauteur max pour les rectangles
    max_pos_y = plot_df['Positive sentiment calculated by VADER'].max()
    y_top = max_pos_y * 1.05 if pd.notna(max_pos_y) and max_pos_y > 0 else 1.0

    # --- ZONE 1 : LE PIÈGE DU SARCASME (Droite / Faux Positif) ---
    fig.add_shape(
        type="rect",
        x0=0.3, y0=0.1, x1=1.05, y1=y_top, # On commence à 0.3 pour attraper les faux positifs
        line=dict(color="Red", width=2, dash="dot"),
        fillcolor="Red",
        opacity=0.08
    )
    fig.add_annotation(
        x=0.80, y=0.5,
        text="<b>⚠️ THE SARCASM TRAP</b><br>Hidden Hostility.<br>VADER thinks this is nice.",
        showarrow=True, arrowhead=2, ax=-60, ay=40,
        bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#d62728",
        font=dict(color="#d62728", size=11, family="Arial Black")
    )

    # --- ZONE 2 : LA ZONE D'HONNÊTETÉ (Gauche / Vrai Négatif) ---
    # C'est la zone où VADER a raison : le texte est méchant, le score est méchant.
    fig.add_shape(
        type="rect",
        x0=-1.05, y0=-0.02, x1=-0.1, y1=y_top,
        line=dict(color="Green", width=2, dash="dot"),
        fillcolor="Green", # Vert car l'algo a "bon" ici
        opacity=0.08
    )
    fig.add_annotation(
        x=-0.5, y=0.5,
        text="<b>✅ THE HONESTY ZONE</b><br>Direct & Explicit Attacks.<br>No sarcasm, just pure negativity.",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        font=dict(color="Green", size=12, family="Arial Black")
    )

    # 4. Finitions
    fig.update_layout(
        xaxis_title="Sentiment Score (⬅️ VADER detects Negativity | VADER detects Positivity ➡️)",
        yaxis_title="Intensity of Positive Words",
        xaxis=dict(range=[-1.05, 1.05]),
        yaxis=dict(range=[-0.02, y_top])
    )

    fig.show()