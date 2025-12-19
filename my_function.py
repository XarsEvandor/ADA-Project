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

import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from plotly.subplots import make_subplots

def apply_dark_theme(fig):
    """Apply war theme styling to any Plotly figure"""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(18,19,26,1)',
        font=dict(color='#e5e7eb', family='Space Grotesk, sans-serif'),
        title_font=dict(color='#e5e7eb'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')),
    )
    # Update axes if they exist
    fig.update_xaxes(gridcolor='#2a2d3a', linecolor='#2a2d3a', tickfont=dict(color='#9ca3af'))
    fig.update_yaxes(gridcolor='#2a2d3a', linecolor='#2a2d3a', tickfont=dict(color='#9ca3af'))
    return fig




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


# Part 2


# Distribution of hyperlinks depending on the semantic distance


def plot_semantic_distance_line_negative(merge_df, sub_df, nbins=50):
    """
    Displays a Line Chart of interaction VOLUME vs. Normalized Semantic Distance.
    Shows ONLY negative hyperlinks to focus on "The Radius of Rivalry".
    """
    
    # 1. Identify embedding columns
    emb_cols = [c for c in sub_df.columns if str(c).startswith('emb_') and c != 'emb_norm']
    
    # 2. Vector Preparation
    subs_vec = sub_df[['SUBREDDIT'] + emb_cols].set_index('SUBREDDIT')
    
    # Filter valid links
    valid_links = merge_df[
        merge_df['SOURCE_SUBREDDIT'].isin(subs_vec.index) & 
        merge_df['TARGET_SUBREDDIT'].isin(subs_vec.index)
    ].copy()
    
    # 3. Vector Calculation
    src_vectors = subs_vec.loc[valid_links['SOURCE_SUBREDDIT']].values
    tgt_vectors = subs_vec.loc[valid_links['TARGET_SUBREDDIT']].values
    
    # Raw Euclidean Distance
    raw_distances = np.linalg.norm(src_vectors - tgt_vectors, axis=1)
    
    # 4. Normalization (0-1)
    min_dist = raw_distances.min()
    max_dist = raw_distances.max()
    
    if max_dist - min_dist == 0:
        valid_links['semantic_distance_norm'] = 0
    else:
        valid_links['semantic_distance_norm'] = (raw_distances - min_dist) / (max_dist - min_dist)
    

    # Filter for ONLY Negative Links (Attacks)
    neg_dist = valid_links[valid_links['LINK_SENTIMENT'] == -1]['semantic_distance_norm']

    # 5. Calculate Bins (np.histogram)
    neg_counts, bin_edges = np.histogram(neg_dist, bins=nbins, range=(0,1))
    
    # Calculate bin centers for the X-axis
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 6. Create Plot (Line Chart)
    fig = go.Figure()

    # Negative Links Line (Red)
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=neg_counts,
        mode='lines',
        name='Negative Links (Attacks)',
        line=dict(color='#EF553B', width=4),
        fill='tozeroy', 
        fillcolor='rgba(239, 85, 59, 0.15)'
    ))

    # 7. Layout
    fig.update_layout(
        # Configuration pour centrer le titre
        title={
            'text': '<b>Neighborly Friction: Mapping the Distance of Discord</b><br>',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Semantic Distance',
        yaxis_title='Number of Hostile Hyperlinks',
        template='plotly_white',
        width=900, 
        height=600,
        legend=dict(x=0.75, y=0.95),
        hovermode="x" 
    )
    
    fig.show()





def evaluate_kmeans_clusters(df, start_k=2, end_k=40, emb_prefix='emb_'):
    """
    Runs K-Means for a range of K, calculates Inertia and Silhouette scores,
    and plots them on a dual-axis chart to help find the optimal K.
    """
    # 1. Prepare Data
    print("Preparing data...")
    # Select columns starting with prefix (e.g., 'emb_') but exclude 'emb_norm'
    emb_cols = [c for c in df.columns if str(c).startswith(emb_prefix) and c != 'emb_norm']
    
    if not emb_cols:
        print(f"Error: No columns found starting with '{emb_prefix}'")
        return
        
    X = df[emb_cols].values
    print(f"Data shape: {X.shape}")

    # 2. Run Clustering Loop
    range_n_clusters = range(start_k, end_k + 1)
    inertias = []
    silhouette_scores = []

    print(f"Calculating Inertia and Silhouette scores for K={start_k} to {end_k}...")

    for k in range_n_clusters:
        # Initialize KMeans (n_init=10 is default but good to be explicit)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Store metrics
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)

    # 3. Plot Results
    print("Plotting results...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary Axis: Inertia (Elbow Method)
    color_inertia = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Lower is better)', color=color_inertia, fontweight='bold')
    ax1.plot(range_n_clusters, inertias, 'o--', color=color_inertia, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color_inertia)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Secondary Axis: Silhouette Score
    ax2 = ax1.twinx()
    color_sil = 'tab:red'
    ax2.set_ylabel('Silhouette Score (Higher is better)', color=color_sil, fontweight='bold')
    ax2.plot(range_n_clusters, silhouette_scores, 's-', color=color_sil, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color_sil)

    # Final Layout
    plt.title('Optimal K Analysis: Elbow Method vs. Silhouette Score')
    # Set x-ticks to show every integer if range is small, or steps if large
    if len(range_n_clusters) < 30:
        plt.xticks(range_n_clusters)
    else:
        plt.xticks(range(start_k, end_k + 1, 2))
        
    fig.tight_layout()
    plt.show()



def run_kmeans_and_analyze(df, k=19, emb_prefix='emb_'):
    """
    Runs K-Means clustering on embedding columns, assigns cluster labels,
    and prints the percentage distribution of subreddits per cluster.
    """
    print(f"--- Running K-Means with K={k} ---")

    # 1. Extract Embeddings
    # Select columns starting with prefix, excluding 'emb_norm' if present
    emb_cols = [c for c in df.columns if str(c).startswith(emb_prefix) and c != 'emb_norm']
    
    if not emb_cols:
        print(f"Error: No columns found starting with '{emb_prefix}'")
        return df
        
    X = df[emb_cols].values
    
    # 2. Fit K-Means
    # n_init=10 is explicit to ensure stability across runs (using K-Means++ logic)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Assign labels to the DataFrame
    # Using .copy() ensures we don't trigger SettingWithCopy warnings if df is a slice
    df_result = df.copy()
    df_result['cluster'] = kmeans.fit_predict(X)
    
    # 3. Analyze & Print Proportions
    print("Proportion of subreddits per cluster (%):")
    print("-" * 45)
    
    # Calculate counts and percentages
    counts = df_result['cluster'].value_counts().sort_index()
    proportions = df_result['cluster'].value_counts(normalize=True).sort_index() * 100
    
    # Formatted output
    for cluster_id, pct in proportions.items():
        count = counts[cluster_id]
        print(f"Cluster {cluster_id:02d}: {pct:6.2f}%  ({count} subreddits)")
        
    print("-" * 45)
    
    # Optional: Quick check for dominant clusters
    max_cluster = proportions.idxmax()
    max_prop = proportions.max()
    if max_prop > 50:
        print(f" Note: Cluster {max_cluster} is very large ({max_prop:.2f}%).")
        print("   Even if it contains >50% of subs, check if its total *activity* is balanced.")
        
    return df_result


def analyze_cluster_contents(sub_df, merge_df, cluster_col='cluster', n_show=20):
    """
    1. Calculates total activity (Source + Target counts) for each subreddit.
    2. Prints the Top N most active subreddits per cluster.
    3. Prints N random subreddits per cluster (to check diversity).
    
    Returns:
        pd.DataFrame: The sub_df updated with an 'activity' column.
    """
    print("--- Calculating Activity Metrics ---")
    
    # 1. Prepare activity data (Sum of outgoing + incoming links)
    # Using .add(..., fill_value=0) ensures we don't get NaN if a sub only exists in one column
    src_counts = merge_df['SOURCE_SUBREDDIT'].value_counts()
    tgt_counts = merge_df['TARGET_SUBREDDIT'].value_counts()
    total_activity = src_counts.add(tgt_counts, fill_value=0)
    
    # Map to the main dataframe
    # We use a copy to avoid SettingWithCopyWarning
    df_result = sub_df.copy()
    df_result['activity'] = df_result['SUBREDDIT'].map(total_activity).fillna(0).astype(int)
    
    # Get list of clusters (sorted)
    if cluster_col not in df_result.columns:
        print(f"Error: Column '{cluster_col}' not found in dataframe.")
        return df_result
        
    clusters = sorted(df_result[cluster_col].unique())
    
    # 2. Print Top Active Subreddits
    print(f"\n=== TOP {n_show} MOST ACTIVE SUBREDDITS PER CLUSTER ===")
    for c in clusters:
        cluster_subs = df_result[df_result[cluster_col] == c]
        # Sort by activity descending
        top_subs = cluster_subs.sort_values('activity', ascending=False).head(n_show)['SUBREDDIT'].tolist()
        print(f"Cluster {c}: {', '.join(top_subs)}")

    # 3. Print Random Subreddits
    print(f"\n=== {n_show} RANDOM SUBREDDITS PER CLUSTER (Diversity Check) ===")
    for c in clusters:
        cluster_subs = df_result[df_result[cluster_col] == c]
        # Sample safely (handle cases where cluster size < n_show)
        n_sample = min(n_show, len(cluster_subs))
        if n_sample > 0:
            random_subs = cluster_subs.sample(n=n_sample, random_state=42)['SUBREDDIT'].tolist()
            print(f"Cluster {c}: {', '.join(random_subs)}")
        else:
            print(f"Cluster {c}: (Empty)")
            
    return df_result




def plot_dual_warfare_matrix(interactions_df, embeddings_df, tribe_mapping, min_interactions=5):
    """
    Plots two matrices side-by-side:
    1. Standard Heatmap: Visualizes the 'Fixation' (Intensity/Ratio) clearly.
    2. Bubble Matrix: Visualizes the 'Volume' (Size) + 'Fixation' (Color).
    """
    # --- 1. Data Prep ---
    sub_to_cluster = embeddings_df.set_index('SUBREDDIT')['cluster'].to_dict()
    
    # Filter for Attacks (Negative Links)
    df_neg = interactions_df[interactions_df['LINK_SENTIMENT'] == -1].copy()
    
    # Map to Tribes
    df_neg['Source_Cluster'] = df_neg['SOURCE_SUBREDDIT'].map(sub_to_cluster)
    df_neg['Target_Cluster'] = df_neg['TARGET_SUBREDDIT'].map(sub_to_cluster)
    df_neg.dropna(subset=['Source_Cluster', 'Target_Cluster'], inplace=True)
    
    df_neg['Source_Tribe'] = df_neg['Source_Cluster'].map(tribe_mapping)
    df_neg['Target_Tribe'] = df_neg['Target_Cluster'].map(tribe_mapping)
    
    # --- 2. Calculate Metrics ---
    # Group by Tribe-to-Tribe
    matrix_df = df_neg.groupby(['Source_Tribe', 'Target_Tribe']).size().reset_index(name='Attack_Volume')
    
    # Calculate Total Outgoing Attacks per Tribe (to normalize for 'Fixation')
    total_outgoing = df_neg.groupby('Source_Tribe').size().to_dict()
    
    # Calculate 'Fixation Score' (% of a Tribe's total attacks directed at this specific target)
    matrix_df['Fixation_Score'] = matrix_df.apply(
        lambda x: (x['Attack_Volume'] / total_outgoing.get(x['Source_Tribe'], 1)) * 100, axis=1
    )
    
    # Filter noise
    matrix_df = matrix_df[matrix_df['Attack_Volume'] >= min_interactions]
    
    # Get sorted list of tribes for axes
    all_tribes = sorted(list(set(tribe_mapping.values())))
    
    # --- 3. Prepare Grid Data ---
    # We need a full grid for the heatmap (pivot table)
    pivot_fixation = matrix_df.pivot(index='Source_Tribe', columns='Target_Tribe', values='Fixation_Score').reindex(index=all_tribes, columns=all_tribes).fillna(0)
    
    # --- 4. Plotting Side-by-Side ---
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.45, 0.55], # Give Bubble chart slightly more space
        subplot_titles=("<b>The Obsession Map</b> (Intensity %)", "<b>The Warfare Matrix</b> (Volume and Intensity)"),
        horizontal_spacing=0.15
    )

    # --- PLOT 1: Regular Heatmap (The "Intent") ---
    # Shows pure Fixation Score (Color)
    fig.add_trace(go.Heatmap(
        z=pivot_fixation.values,
        x=all_tribes,
        y=all_tribes,
        colorscale='Reds',
        colorbar=dict(title="Fixation %", x=0.42, len=0.8), # Place colorbar in middle
        hovertemplate='Source: %{y}<br>Target: %{x}<br>Fixation: %{z:.1f}%<extra></extra>'
    ), row=1, col=1)

    # --- PLOT 2: Bubble Matrix (The "Impact") ---
    # Size = Volume, Color = Fixation Score
    
    # Normalize size for bubbles
    max_vol = matrix_df['Attack_Volume'].max()
    sizeref = 2.0 * max_vol / (30**2) # Scaling factor
    
    fig.add_trace(go.Scatter(
        x=matrix_df['Target_Tribe'],
        y=matrix_df['Source_Tribe'],
        mode='markers',
        marker=dict(
            size=matrix_df['Attack_Volume'],
            sizemode='area',
            sizeref=sizeref,
            sizemin=3,
            color=matrix_df['Fixation_Score'], # Color matches the heatmap
            colorscale='Reds',
            showscale=False, # Shared color scale
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hovertemplate='<b>Aggressor:</b> %{y}<br><b>Victim:</b> %{x}<br>Volume: %{marker.size}<br>Fixation: %{marker.color:.1f}%<extra></extra>'
    ), row=1, col=2)

    # --- 5. Layout ---
    fig.update_layout(
        title_text="<b>The Tribal Warfare Matrix: Intent vs. Impact</b>",
        height=700,
        width=1300,
        template='plotly_white',
        showlegend=False
    )
    
    # Update Axes
    fig.update_xaxes(title_text="Victim Tribe", tickangle=45, row=1, col=1)
    fig.update_yaxes(title_text="Aggressor Tribe", row=1, col=1, autorange='reversed') # Reverse Y for matrix logic
    
    fig.update_xaxes(title_text="Victim Tribe", tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text="Aggressor Tribe", row=1, col=2, showticklabels=False, autorange='reversed') # Hide Y labels on right plot to save space

    fig.show()




def create_tribe_intelligence_df(sub_merged_df, merge_df, tribe_mapping):
    """
    Consolidates cluster statistics and inter-tribe conflict data into a 
    master 'Tribe Intelligence' DataFrame.
    """
    
    # 1. Calculate Cluster Stats (Size & Activity)
    tribe_df = sub_merged_df.groupby('cluster').agg(
        cluster_size=('SUBREDDIT', 'count'),
        cluster_activity=('activity', 'sum') 
    ).reset_index()

    # Add readable tribe names
    tribe_df['cluster_name'] = tribe_df['cluster'].map(tribe_mapping)

    # 2. Map Clusters to Interactions
    # Create local copies to avoid SettingWithCopy warnings
    interactions = merge_df.copy()
    sub_to_cluster = sub_merged_df.set_index('SUBREDDIT')['cluster'].to_dict()
    
    interactions['source_cluster'] = interactions['SOURCE_SUBREDDIT'].map(sub_to_cluster)
    interactions['target_cluster'] = interactions['TARGET_SUBREDDIT'].map(sub_to_cluster)

    # 3. Analyze Inter-Cluster Conflict (Negative Links Only)
    neg_links = interactions[interactions['LINK_SENTIMENT'] == -1]
    attack_counts = neg_links.groupby(['source_cluster', 'target_cluster']).size().reset_index(name='count')

    # 4. Identify Top 3 Rivals for Each Tribe
    top_attackers_num, top_victims_num = [], []
    top_attackers_name, top_victims_name = [], []

    sorted_clusters = sorted(tribe_df['cluster'].unique())

    for cluster_id in sorted_clusters:
        # --- Incoming: Who attacks THIS tribe? ---
        incoming = attack_counts[attack_counts['target_cluster'] == cluster_id]
        top_inc = incoming.sort_values('count', ascending=False).head(3)
        
        inc_ids = top_inc['source_cluster'].astype(int).tolist()
        inc_names = [tribe_mapping.get(i, f"Cluster {i}") for i in inc_ids]
        
        # --- Outgoing: Who does THIS tribe attack? ---
        outgoing = attack_counts[attack_counts['source_cluster'] == cluster_id]
        top_out = outgoing.sort_values('count', ascending=False).head(3)
        
        out_ids = top_out['target_cluster'].astype(int).tolist()
        out_names = [tribe_mapping.get(i, f"Cluster {i}") for i in out_ids]
        
        # Store intelligence
        top_attackers_num.append(inc_ids)
        top_victims_num.append(out_ids)
        top_attackers_name.append(inc_names)
        top_victims_name.append(out_names)

    # 5. Assemble and Format the Intelligence Ledger
    tribe_df['top_3_attackers_num'] = top_attackers_num
    tribe_df['top_3_victims_num'] = top_victims_num
    tribe_df['top_3_attackers_name'] = top_attackers_name
    tribe_df['top_3_victims_name'] = top_victims_name

    # Clean up column names for the Guide
    tribe_df = tribe_df.rename(columns={'cluster': 'cluster_number'})
    
    cols = [
        'cluster_number', 'cluster_name', 'cluster_size', 'cluster_activity',
        'top_3_attackers_num', 'top_3_victims_num', 
        'top_3_attackers_name', 'top_3_victims_name'
    ]
    
    return tribe_df[cols]




def plot_interactive_tribe_radar(interactions_df, embeddings_df, tribe_mapping, threshold=50):
    """
    Generates an Interactive Radar Chart with Dynamic Axes.
    FIXED: Increased margins to prevent long Tribe names from being cut off.
    """
    # 1. Prepare Data
    if 'cluster' not in embeddings_df.columns:
        print("Error: 'cluster' column missing in embeddings_df.")
        return

    # --- Calculate Tribe Stats (Size & Activity) ---
    stats_df = embeddings_df.copy()
    if 'activity' not in stats_df.columns:
        act = interactions_df['SOURCE_SUBREDDIT'].value_counts().add(
            interactions_df['TARGET_SUBREDDIT'].value_counts(), fill_value=0
        )
        stats_df['activity'] = stats_df['SUBREDDIT'].map(act).fillna(0)
    
    stats_df['tribe'] = stats_df['cluster'].map(tribe_mapping)
    
    tribe_stats = stats_df.groupby('tribe').agg(
        tribe_size=('SUBREDDIT', 'count'),
        tribe_activity=('activity', 'sum')
    )

    # --- Prepare Radar Data ---
    sub_to_cluster = embeddings_df.set_index('SUBREDDIT')['cluster'].to_dict()
    df_neg = interactions_df[interactions_df['LINK_SENTIMENT'] == -1].copy()
    
    df_neg['Source_Cluster'] = df_neg['SOURCE_SUBREDDIT'].map(sub_to_cluster)
    df_neg['Target_Cluster'] = df_neg['TARGET_SUBREDDIT'].map(sub_to_cluster)
    df_neg.dropna(subset=['Source_Cluster', 'Target_Cluster'], inplace=True)
    
    df_neg['Source_Tribe'] = df_neg['Source_Cluster'].map(tribe_mapping)
    df_neg['Target_Tribe'] = df_neg['Target_Cluster'].map(tribe_mapping)
    
    all_tribes = sorted(list(set(tribe_mapping.values())))
    
    # 2. Pre-calculate Data for Each Tribe
    tribe_data_store = {}
    
    for focus_tribe in all_tribes:
        relevant_df = df_neg[
            (df_neg['Source_Tribe'] == focus_tribe) | 
            (df_neg['Target_Tribe'] == focus_tribe)
        ]
        
        in_counts = relevant_df[relevant_df['Target_Tribe'] == focus_tribe]['Source_Tribe'].value_counts()
        out_counts = relevant_df[relevant_df['Source_Tribe'] == focus_tribe]['Target_Tribe'].value_counts()
        
        total_vol = in_counts.add(out_counts, fill_value=0)
        significant_rivals = total_vol[total_vol >= threshold].index.tolist()
        significant_rivals = sorted(significant_rivals)
        
        if not significant_rivals:
            tribe_data_store[focus_tribe] = {'theta': ['None'], 'r_in': [0], 'r_out': [0], 'max': 1}
            continue

        r_in = [in_counts.get(rival, 0) for rival in significant_rivals]
        r_out = [out_counts.get(rival, 0) for rival in significant_rivals]
        
        theta_final = significant_rivals + [significant_rivals[0]]
        r_in_final = r_in + [r_in[0]]
        r_out_final = r_out + [r_out[0]]
        
        local_max = max(max(r_in), max(r_out))
        if local_max == 0: local_max = 1
        
        tribe_data_store[focus_tribe] = {
            'theta': theta_final,
            'r_in': r_in_final,
            'r_out': r_out_final,
            'max': local_max
        }

    # 3. Initialize Figure
    first_tribe = all_tribes[0]
    first_data = tribe_data_store[first_tribe]
    s_size = int(tribe_stats.loc[first_tribe, 'tribe_size'])
    s_act = int(tribe_stats.loc[first_tribe, 'tribe_activity'])
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=first_data['r_in'], theta=first_data['theta'], fill='toself',
        name='Attacked BY (Incoming)',
        line_color='rgba(128, 0, 128, 0.8)', fillcolor='rgba(128, 0, 128, 0.2)', marker=dict(size=4)
    ))

    fig.add_trace(go.Scatterpolar(
        r=first_data['r_out'], theta=first_data['theta'], fill='toself',
        name='Attacks TO (Outgoing)',
        line_color='rgba(255, 0, 0, 0.8)', fillcolor='rgba(255, 0, 0, 0.2)', marker=dict(size=4)
    ))

    # 4. Create Dropdown
    buttons = []
    for tribe in all_tribes:
        data = tribe_data_store[tribe]
        t_size = int(tribe_stats.loc[tribe, 'tribe_size']) if tribe in tribe_stats.index else 0
        t_act = int(tribe_stats.loc[tribe, 'tribe_activity']) if tribe in tribe_stats.index else 0
        
        new_title = (
            f"Strategic Radar: <b>{tribe}</b><br>"
            f"<span style='font-size:14px; color:#555;'>Size: {t_size} subs | Activity: {t_act} links</span>"
        )
        
        buttons.append(dict(
            label=tribe, method="update", 
            args=[
                {"r": [data['r_in'], data['r_out']], "theta": [data['theta'], data['theta']]},
                {"title": new_title, "polar.radialaxis.range": [0, data['max'] * 1.1]}
            ]
        ))

    # 5. Layout (WITH IMPROVED MARGINS)
    init_title = (
        f"Strategic Radar: <b>{first_tribe}</b><br>"
        f"<span style='font-size:14px; color:#555;'>Size: {s_size} subs | Activity: {s_act} links</span>"
    )

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=1.2, y=1, xanchor='left', yanchor='top')],
        polar=dict(
            radialaxis=dict(visible=True, range=[0, first_data['max'] * 1.1]),
        ),
        title=dict(text=init_title, y=0.95, x=0.5, xanchor='center', font=dict(size=20)),
        legend=dict(x=0.5, y=-0.15, xanchor='center', orientation='h'),
        height=800, 
        width=1100, # Increased Width
        # --- KEY FIX ---
        # Large margins on Left/Right to accommodate long text labels (e.g., "Internet Culture...")
        margin=dict(t=120, b=100, l=200, r=200) 
    )

    fig.show()



def generate_interactive_ballistics_report(interactions_df, embeddings_df, tribe_mapping, target_list):
    """
    Regroups all Sankey logic into a single interactive tool.
    Analyzes the flow of hostility for a specific list of High-Value Targets.
    """
    
    # --- Internal Helper: Process individual subreddit data ---
    def get_sankey_data(target_sub):
        # 1. Get Tribe Name
        target_cluster = embeddings_df[embeddings_df['SUBREDDIT'] == target_sub]['cluster'].values
        tribe_name = tribe_mapping.get(target_cluster[0], "Unknown Tribe") if len(target_cluster) > 0 else "Unknown Tribe"

        # 2. Calculate High-Level Stats
        all_involving = interactions_df[
            (interactions_df['SOURCE_SUBREDDIT'] == target_sub) | 
            (interactions_df['TARGET_SUBREDDIT'] == target_sub)
        ]
        
        if all_involving.empty:
            return None

        total_activity = len(all_involving)
        incoming_attacks = len(interactions_df[(interactions_df['TARGET_SUBREDDIT'] == target_sub) & (interactions_df['LINK_SENTIMENT'] == -1)])
        outgoing_attacks = len(interactions_df[(interactions_df['SOURCE_SUBREDDIT'] == target_sub) & (interactions_df['LINK_SENTIMENT'] == -1)])

        # 3. Prepare Flow Data
        sub_to_cluster = embeddings_df.set_index('SUBREDDIT')['cluster'].to_dict()
        
        # Incoming Attacks (Purple)
        in_df = interactions_df[(interactions_df['TARGET_SUBREDDIT'] == target_sub) & (interactions_df['LINK_SENTIMENT'] == -1)].copy()
        in_df['tribe'] = in_df['SOURCE_SUBREDDIT'].map(sub_to_cluster).map(tribe_mapping)
        in_counts = in_df['tribe'].value_counts().reset_index().head(10)
        in_counts.columns = ['tribe', 'count']
        
        # Outgoing Attacks (Red)
        out_df = interactions_df[(interactions_df['SOURCE_SUBREDDIT'] == target_sub) & (interactions_df['LINK_SENTIMENT'] == -1)].copy()
        out_df['tribe'] = out_df['TARGET_SUBREDDIT'].map(sub_to_cluster).map(tribe_mapping)
        out_counts = out_df['tribe'].value_counts().reset_index().head(10)
        out_counts.columns = ['tribe', 'count']

        # 4. Define Nodes & Indices
        unique_tribes_in = in_counts['tribe'].unique().tolist()
        unique_tribes_out = out_counts['tribe'].unique().tolist()
        label_list = unique_tribes_in + [f"r/{target_sub}"] + unique_tribes_out
        target_idx = len(unique_tribes_in) 
        
        sources, targets, values, colors = [], [], [], []
        
        # Map Purple Links (Left to Center)
        for _, row in in_counts.iterrows():
            sources.append(unique_tribes_in.index(row['tribe']))
            targets.append(target_idx)
            values.append(row['count'])
            colors.append("rgba(128, 0, 128, 0.6)")
            
        # Map Red Links (Center to Right)
        for _, row in out_counts.iterrows():
            sources.append(target_idx)
            targets.append(target_idx + 1 + unique_tribes_out.index(row['tribe']))
            values.append(row['count'])
            colors.append("rgba(255, 0, 0, 0.6)")

        return {
            'node': dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=label_list, color="darkgray"),
            'link': dict(source=sources, target=targets, value=values, color=colors),
            'title': (f"<b>Flow of Hostility: r/{target_sub}</b> ({tribe_name})<br>"
                      f"<span style='font-size:14px;'>Total Activity: <b>{total_activity}</b> | "
                      f"<span style='color: purple;'>Incoming: <b>{incoming_attacks}</b></span> | "
                      f"<span style='color: red;'>Outgoing: <b>{outgoing_attacks}</b></span></span>")
        }

    # --- Main Execution ---
    fig = go.Figure()
    buttons = []
    
    # Initialize with the first valid entry in the list
    first_data = None
    for sub in target_list:
        data = get_sankey_data(sub)
        if data:
            if first_data is None:
                first_data = data
                fig.add_trace(go.Sankey(node=data['node'], link=data['link']))
            
            # Add to dropdown
            buttons.append(dict(
                label=f"r/{sub}",
                method="update",
                args=[{"node": [data['node']], "link": [data['link']]}, {"title": data['title']}]
            ))

    # Layout styling
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=1.1, y=1, xanchor='left', yanchor='top')],
        title_text=first_data['title'] if first_data else "No Data Found",
        height=700, width=1200,
        margin=dict(t=120, b=20, l=20, r=250),
        template='plotly_white'
    )

    fig.show()



def plot_top_sarcastic_subreddits(df, min_attacks=50, top_n=20):
    """
    Identifie et visualise les subreddits qui utilisent le plus le sarcasme comme arme.
    
    Critères :
    - On ne garde que les communautés ayant lancé au moins 'min_attacks' attaques (pour éviter le bruit).
    - On classe par le % d'attaques qui sont positives selon VADER (Faux Positifs).
    """
    print("Identification des 'Rois du Sarcasme'...")

    # 1. Filtrage : On ne garde que les attaques réelles (LINK_SENTIMENT == -1)
    attacks = df[df['LINK_SENTIMENT'] == -1].copy()

    # 2. Détection du Sarcasme (Attaque mais Texte Positif)
    # Seuil : Compound >= 0.3 et Positive >= 0.1 (Critères stricts du sarcasme)
    attacks['is_sarcasm'] = (
        (attacks['Compound sentiment calculated by VADER'] >= 0.3) & 
        (attacks['Positive sentiment calculated by VADER'] >= 0.1)
    ).astype(int)

    # 3. Agrégation
    stats = attacks.groupby('SOURCE_SUBREDDIT').agg(
        total_attacks=('is_sarcasm', 'count'),
        sarcastic_count=('is_sarcasm', 'sum')
    ).reset_index()

    # 4. Filtrage de robustesse (On ignore les petits subs)
    stats = stats[stats['total_attacks'] >= min_attacks].copy()

    if len(stats) == 0:
        print(f"⚠️ Aucun subreddit avec plus de {min_attacks} attaques trouvé.")
        return

    # 5. Calcul du Ratio
    stats['sarcasm_ratio'] = (stats['sarcastic_count'] / stats['total_attacks']) * 100
    
    # 6. Tri et Sélection du Top N
    top_sarcastic = stats.sort_values('sarcasm_ratio', ascending=True).tail(top_n)

    # 7. Plot (Bar Chart Horizontal)
    fig = px.bar(
        top_sarcastic,
        x="sarcasm_ratio",
        y="SOURCE_SUBREDDIT",
        orientation='h',
        title=f"<b>The Sarcasm Elite:</b> Top {top_n} Subreddits masking hostility as praise",
        labels={"sarcasm_ratio": "Sarcasm Rate (%)", "SOURCE_SUBREDDIT": ""},
        text=top_sarcastic['sarcasm_ratio'].apply(lambda x: f"{x:.1f}%"),
        template="plotly_white",
        color="sarcasm_ratio",
        color_continuous_scale="Oranges" # Orange pour rappeler le danger caché
    )

    # Styling
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=600 + (top_n * 10),
        xaxis_title="Percentage of Attacks detected as 'Positive' by AI",
        xaxis=dict(range=[0, 100]), # Échelle fixe 0-100%
        coloraxis_showscale=False
    )
    
    # Annotation explicative
    fig.add_annotation(
        x=95, y=0,
        text="<b>HIGH RISK ZONE</b><br>Bots will miss these attacks.<br>Human review required.",
        showarrow=False,
        xref="x", yref="paper",
        font=dict(color="orange", size=12),
        align="right"
    )

    fig.show()



def plot_temporal_toxicity_heatmap(df, target_sentiment=-1, title_suffix=""):
    """
    Génère une Heatmap (Jours vs Heures) pour visualiser les moments chauds de la toxicité.
    
    Args:
        df: Le DataFrame fusionné (merge_df).
        target_sentiment: -1 pour les attaques/sarcasmes, 1 pour le positif.
        title_suffix: Texte optionnel pour le titre (ex: " - Subreddit: r/Politics").
    """
    print("Construction de la Heatmap Temporelle...")

    # 1. Préparation et Filtrage
    # On ne garde que les interactions avec le sentiment cible (ex: -1 pour toxique)
    target_df = df[df['LINK_SENTIMENT'] == target_sentiment].copy()
    
    if not pd.api.types.is_datetime64_any_dtype(target_df['TIMESTAMP']):
        target_df['TIMESTAMP'] = pd.to_datetime(target_df['TIMESTAMP'], utc=True).dt.tz_localize(None)
    
    # Extraction des composants temporels
    target_df['DayOfWeek'] = target_df['TIMESTAMP'].dt.day_name()
    target_df['Hour'] = target_df['TIMESTAMP'].dt.hour
    
    # 2. Création de la Matrice (Pivot Table)
    # On compte le nombre d'attaques par créneau (Jour, Heure)
    heatmap_data = target_df.groupby(['Hour', 'DayOfWeek']).size().unstack(fill_value=0)
    
    # 3. Réorganisation Cruciale
    # Pour avoir l'ordre Lundi -> Dimanche (sinon c'est alphabétique : Friday, Monday...)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # On s'assure que toutes les colonnes existent (même si un jour est vide)
    for day in days_order:
        if day not in heatmap_data.columns:
            heatmap_data[day] = 0
            
    # On réordonne les colonnes et on remplit les heures manquantes (0-23)
    heatmap_data = heatmap_data[days_order]
    heatmap_data = heatmap_data.reindex(range(24), fill_value=0)

    # 4. Plotting avec Plotly Express (imshow est optimisé pour les heatmaps)
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Day of Week", y="Hour of Day", color="Attack Count"),
        x=days_order,
        y=[f"{h:02d}:00" for h in range(24)], # Formatage joli "09:00"
        color_continuous_scale="Reds" if target_sentiment == -1 else "Blues",
        aspect="auto", # Permet aux carrés de s'adapter à la taille de l'écran
        title=f"<b>Temporal Toxicity Pattern</b>: When do they attack? {title_suffix}"
    )

    # 5. Styling "Pro"
    fig.update_layout(
        template="plotly_white",
        height=600,
        xaxis_title="",
        yaxis_title="Time (UTC)",
        # Ajout d'un petit espace entre les tuiles pour la lisibilité
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"), # 00h en haut, 23h en bas
    )
    
    # Ajout des valeurs textuelles dans les cases si la grille n'est pas trop chargée
    # (Uniquement si on regarde un sous-ensemble, sinon c'est illisible)
    if heatmap_data.max().max() < 1000:
        fig.update_traces(text=heatmap_data.values, texttemplate="%{text}")

    fig.show()




def plot_simple_vector(df, vector_type):
    """
    df: Le DataFrame (title_df ou body_df)
    vector_type: 'title' ou 'body' (pour choisir le bon titre narratif)
    """
    
    # 1. Configuration Narrative (Billboard vs Bunker)
    if vector_type.lower() == 'title':
        plot_title = "<b>THE BILLBOARD</b> (Titles)"
        plot_subtitle = "<i>Broadcast Weapon • High Visibility</i>"
    else: # body
        plot_title = "<b>THE BUNKER</b> (Body)"
        plot_subtitle = "<i>Contained Toxicity • Active Engagement</i>"

    # 2. Calcul des Données
    # On filtre les sentiments
    neg_count = len(df[df['LINK_SENTIMENT'] == -1])
    pos_count = len(df[df['LINK_SENTIMENT'] == 1])
    total = neg_count + pos_count
    
    # Calcul du % d'attaque pour l'affichage sur la partie rouge
    attack_pct = (neg_count / total * 100) if total > 0 else 0

    # 3. Création du Donut
    # Rouge pour l'attaque, Gris très pâle pour le reste
    if(vector_type == 'BODY'): 
        colors = ['#8b0000', '#f2f2f2'] 
    else: 
        colors = ['#d62728', '#f2f2f2'] 

    fig = go.Figure(data=[go.Pie(
        labels=['Attacks', 'Safe Context'],
        values=[neg_count, pos_count],
        marker_colors=colors,
        hole=0.6, # Trou large pour l'élégance
        
        # Affichage sélectif du texte :
        # On affiche le % uniquement sur la partie Rouge. Rien sur le gris.
        text=[f"{attack_pct:.1f}%", ""], 
        textinfo='text', 
        textfont_size=20,
        textfont_color='white',
        
        # Le survol (hover) garde les infos détaillées si besoin
        hoverinfo='label+percent',
        sort=False
    )])

    # 4. Mise en page
    fig.update_layout(
        # Titre simple avec le sous-titre narratif
        title_text=f"{plot_title}<br><span style='font-size:14px; color:grey'>{plot_subtitle}</span>",
        title_x=0.5,
        showlegend=False,
        height=400,
        margin=dict(t=80, b=20, l=20, r=20),
        font=dict(family="Arial")
    )

    fig.show()


def plot_attack_origin_comparison(title_df, body_df):
    """
    Compare le volume ABSOLU de liens négatifs entre Title et Body.
    Permet de voir d'où provient la majorité de la toxicité.
    """
    
    # 1. Extraction des volumes d'attaques uniquement
    title_attacks = len(title_df[title_df['LINK_SENTIMENT'] == -1])
    body_attacks = len(body_df[body_df['LINK_SENTIMENT'] == -1])
    
    total_attacks = title_attacks + body_attacks

    # 2. Configuration Visuelle Narrative
    # Rouge Vif pour le Billboard (Visible), Rouge Sombre pour le Bunker (Enfoui)
    colors = ['#ff4d4d', '#8b0000'] 
    
    labels = ['THE BILLBOARD (Titles)', 'THE BUNKER (Body)']
    values = [title_attacks, body_attacks]

    # 3. Création du Graphique
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.5,
        
        # On affiche le Label et le Pourcentage clairement
        textinfo='percent+label',
        textfont_size=14,
        textfont_color='white', # Blanc pour bien ressortir sur le rouge
        
        hoverinfo='label+value+percent', # Au survol, on voit le nombre exact
        sort=False
    )])

    # 4. Mise en page
    fig.update_layout(
        title_text=f"<b>ORIGIN OF HOSTILITY</b><br><i>Distribution of the {total_attacks} detected attacks</i>",
        title_x=0.5,
        showlegend=False, # Pas besoin de légende, tout est écrit sur le camembert
        height=450,
        font=dict(family="Arial"),
        margin=dict(t=80, b=20, l=20, r=20)
    )

    fig.show()


def plot_spike_cascades(
    df,
    source_col="SOURCE_SUBREDDIT",
    target_col="TARGET_SUBREDDIT",
    sentiment_col="LINK_SENTIMENT",
    time_col="TIMESTAMP",
    negative_value=-1,
    sigma=2,
    delta_days=30
):
    # --- 1. FILTER NEGATIVE LINKS ---
    attacks = df[df[sentiment_col] == negative_value].copy()
    attacks["date"] = pd.to_datetime(attacks[time_col])
    attacks["day"] = attacks["date"].dt.date

    # Count INCOMING attacks per day
    daily_in = (
        attacks.groupby([target_col, "day"])
        .size()
        .reset_index(name="n_in")
    )

    # Count OUTGOING attacks per day
    daily_out = (
        attacks.groupby([source_col, "day"])
        .size()
        .reset_index(name="n_out")
    )

    # --- 2. CALCULATE BASELINES ---
    def get_stats(df_, col_sub, col_val):
        stats = df_.groupby(col_sub)[col_val].agg(["mean", "std"]).fillna(0)
        stats["threshold"] = stats["mean"] + sigma * stats["std"]
        return stats

    stats_in = get_stats(daily_in, target_col, "n_in")
    stats_out = get_stats(daily_out, source_col, "n_out")

    daily_in = daily_in.merge(
        stats_in[["threshold"]],
        left_on=target_col,
        right_index=True
    )

    daily_out = daily_out.merge(
        stats_out[["threshold"]],
        left_on=source_col,
        right_index=True
    )

    # --- 3. DETECT SPIKES ---
    daily_in["is_spike_in"] = daily_in["n_in"] > daily_in["threshold"]
    daily_out["is_spike_out"] = daily_out["n_out"] > daily_out["threshold"]

    spikes_in = daily_in[daily_in["is_spike_in"]].copy()
    spikes_out = daily_out[daily_out["is_spike_out"]].copy()

    # --- 4. DETECT CASCADES ---
    cascades = []

    out_lookup = spikes_out.groupby(source_col)

    for _, row_in in spikes_in.iterrows():
        sub = row_in[target_col]
        date_in = row_in["day"]

        if sub in out_lookup.groups:
            potential_reactions = out_lookup.get_group(sub)

            reaction_window = [
                date_in + pd.Timedelta(days=i)
                for i in range(1, delta_days + 1)
            ]

            matches = potential_reactions[
                potential_reactions["day"].isin(reaction_window)
            ]

            for _, row_out in matches.iterrows():
                sources_of_incoming = attacks[
                    (attacks[target_col] == sub) &
                    (attacks["day"] == date_in)
                ][source_col].unique()

                targets_of_outgoing = attacks[
                    (attacks[source_col] == sub) &
                    (attacks["day"] == row_out["day"])
                ][target_col].unique()

                new_targets = np.setdiff1d(
                    targets_of_outgoing,
                    sources_of_incoming
                )

                if len(new_targets) > 0:
                    cascades.append({
                        "SUBREDDIT": sub,
                        "date_trigger": date_in,
                        "date_reaction": row_out["day"],
                        "incoming_vol": row_in["n_in"],
                        "outgoing_vol": row_out["n_out"],
                        "new_victims": list(new_targets)
                    })

    cascade_df = pd.DataFrame(cascades)

    # --- 5. VISUALIZATION ---
    if not cascade_df.empty:
        top_contagious = (
            cascade_df["SUBREDDIT"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_contagious.columns = ["SUBREDDIT", "cascade_count"]

        fig = px.bar(
            top_contagious,
            x="cascade_count",
            y="SUBREDDIT",
            orientation="h",
            title="<b>The Spreaders</b>: SUBREDDITs most likely to lash out after being hit",
            color="cascade_count",
            template="plotly_dark"
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )
        fig.show()

        display(cascade_df.head())
    else:
        print("No cascades found with current thresholds. Try lowering sigma.")

def plot_in_out_over_time(
    df,
    subreddit_name,
    source_col="SOURCE_SUBREDDIT",
    target_col="TARGET_SUBREDDIT",
    sentiment_col="LINK_SENTIMENT",
    time_col="TIMESTAMP",
    negative_value=-1,
    freq="D",          # "D"=daily, "W"=weekly, "M"=monthly
    smooth=None        # None or int window for rolling mean (e.g., 7 for 7-day)
):
    """
    Plots incoming vs outgoing NEGATIVE links over time for a given subreddit.
    Does not return anything (just shows the plot).
    """

    # Filter negative links only
    attacks = df[df[sentiment_col] == negative_value].copy()
    attacks[time_col] = pd.to_datetime(attacks[time_col], errors="coerce")
    attacks = attacks.dropna(subset=[time_col])

    # Bucket time
    attacks["t"] = attacks[time_col].dt.to_period(freq).dt.to_timestamp()

    # Incoming (target perspective)
    incoming_ts = (
        attacks[attacks[target_col] == subreddit_name]
        .groupby("t")
        .size()
        .rename("incoming")
    )

    # Outgoing (source perspective)
    outgoing_ts = (
        attacks[attacks[source_col] == subreddit_name]
        .groupby("t")
        .size()
        .rename("outgoing")
    )

    # Align on same time index (fill missing with 0)
    ts = pd.concat([incoming_ts, outgoing_ts], axis=1).fillna(0).sort_index()

    # Optional smoothing (rolling mean)
    if smooth is not None and smooth > 1:
        ts_plot = ts.rolling(window=smooth, min_periods=1).mean()
        title_suffix = f" (rolling mean, window={smooth})"
    else:
        ts_plot = ts
        title_suffix = ""

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_plot.index, y=ts_plot["incoming"],
        mode="lines",
        name="Incoming (targeted)"
    ))
    fig.add_trace(go.Scatter(
        x=ts_plot.index, y=ts_plot["outgoing"],
        mode="lines",
        name="Outgoing (attacks)"
    ))

    fig.update_layout(
        title=f"Incoming vs Outgoing Negative Links Over Time — r/{subreddit_name}{title_suffix}",
        xaxis_title="Time",
        yaxis_title=f"Count per {freq}",
        template="plotly_dark"
    )

    fig.show()

def plot_outgoing_pos_neg_over_time(
    df,
    subreddit_name,
    source_col="SOURCE_SUBREDDIT",
    sentiment_col="LINK_SENTIMENT",
    time_col="TIMESTAMP",
    positive_value=1,
    negative_value=-1,
    freq="D",      # "D" daily, "W" weekly, "M" monthly
    smooth=None    # e.g. 7 for 7-day rolling mean
):
    """
    Plots outgoing POSITIVE vs NEGATIVE links over time for a given subreddit.
    Shows one curve for positives, one for negatives.
    Does not return anything.
    """

    # Filter to outgoing links of the subreddit
    df_sub = df[df[source_col] == subreddit_name].copy()
    df_sub[time_col] = pd.to_datetime(df_sub[time_col], errors="coerce")
    df_sub = df_sub.dropna(subset=[time_col])

    # Time binning
    df_sub["t"] = df_sub[time_col].dt.to_period(freq).dt.to_timestamp()

    # Count outgoing NEGATIVE links
    neg_ts = (
        df_sub[df_sub[sentiment_col] == negative_value]
        .groupby("t")
        .size()
        .rename("negative")
    )

    # Count outgoing POSITIVE links
    pos_ts = (
        df_sub[df_sub[sentiment_col] == positive_value]
        .groupby("t")
        .size()
        .rename("positive")
    )

    # Align time index
    ts = pd.concat([neg_ts, pos_ts], axis=1).fillna(0).sort_index()

    # Optional smoothing
    if smooth is not None and smooth > 1:
        ts_plot = ts.rolling(window=smooth, min_periods=1).mean()
        title_suffix = f" (rolling mean, window={smooth})"
    else:
        ts_plot = ts
        title_suffix = ""

    # Plot
    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=ts_plot.index,
        y=ts_plot["positive"],
        mode="lines",
        name="Outgoing positive links"
    ))

    fig.add_trace(go.Scatter(
        x=ts_plot.index,
        y=ts_plot["negative"],
        mode="lines",
        name="Outgoing negative links"
    ))


    fig.update_layout(
        title=f"Outgoing Positive vs Negative Links Over Time — r/{subreddit_name}{title_suffix}",
        xaxis_title="Time",
        yaxis_title=f"Count per {freq}",
        template="plotly_dark"
    )

    fig.show()

def plot_subreddit_connections_2d_overtime(
    df,
    subreddit_name,
    source_col="SOURCE_SUBREDDIT",
    target_col="TARGET_SUBREDDIT",
    sx="source_x",
    sy="source_y",
    tx="target_x",
    ty="target_y",
    time_col="TIMESTAMP",
    freq="W",                # "D", "W", "M"
    sentiment_col="LINK_SENTIMENT",      # e.g. "LINK_SENTIMENT"
    sentiment_value=None,    # e.g. -1 or +1
    max_edges_per_frame=300,
    show_labels=False,
    template="plotly_dark"
):
    """
    Animated 2D map of connections involving `subreddit_name`, over time (slider).
    Does not return anything.
    """

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])

    d = d[d[time_col] >= pd.Timestamp("2014-01-01", tz="UTC")]

    # Optional sentiment filtering
    if sentiment_col is not None and sentiment_value is not None:
        d = d[d[sentiment_col] == sentiment_value]

    # Keep only edges involving the subreddit
    mask = (d[source_col] == subreddit_name) | (d[target_col] == subreddit_name)
    d = d[mask].copy()
    if d.empty:
        print(f"No edges found for '{subreddit_name}' (with current filters).")
        return

    # Period binning
    d["period"] = d[time_col].dt.to_period(freq).dt.to_timestamp()

    # Ensure we have target coords; if missing, infer from SOURCE coords across whole df
    if tx not in d.columns or ty not in d.columns:
        coord_map = (
            df[[source_col, sx, sy]]
            .dropna()
            .drop_duplicates(subset=[source_col])
            .set_index(source_col)[[sx, sy]]
        )
        d = d.join(coord_map, on=target_col, rsuffix="_t")
        d[tx] = d.get(f"{sx}_t")
        d[ty] = d.get(f"{sy}_t")

    d = d.dropna(subset=[sx, sy, tx, ty]).copy()
    if d.empty:
        print(f"Missing endpoint coordinates. Need '{tx}'/'{ty}' or inferable coords.")
        return

    # Helper to build line segments
    def make_lines(edges, x0, y0, x1, y1):
        xs, ys = [], []
        for a, b, c, d_ in zip(edges[x0], edges[y0], edges[x1], edges[y1]):
            xs += [a, c, None]
            ys += [b, d_, None]
        return xs, ys

    periods = sorted(d["period"].unique())

    # Precompute global axis bounds so the map doesn't jump around
    x_min = float(np.nanmin(pd.concat([d[sx], d[tx]])))
    x_max = float(np.nanmax(pd.concat([d[sx], d[tx]])))
    y_min = float(np.nanmin(pd.concat([d[sy], d[ty]])))
    y_max = float(np.nanmax(pd.concat([d[sy], d[ty]])))

    # Build frames
    frames = []
    for p in periods:
        dp = d[d["period"] == p].copy()

        out_edges = dp[dp[source_col] == subreddit_name].copy()
        in_edges  = dp[dp[target_col] == subreddit_name].copy()

        # cap edges per frame for readability
        if len(out_edges) > max_edges_per_frame:
            out_edges = out_edges.sample(max_edges_per_frame, random_state=0)
        if len(in_edges) > max_edges_per_frame:
            in_edges = in_edges.sample(max_edges_per_frame, random_state=0)

        out_xs, out_ys = make_lines(out_edges, sx, sy, tx, ty)
        in_xs,  in_ys  = make_lines(in_edges,  sx, sy, tx, ty)

        # nodes in this period
        nodes = pd.concat([
            out_edges[[source_col, sx, sy]].rename(columns={source_col:"sub", sx:"x", sy:"y"}),
            out_edges[[target_col, tx, ty]].rename(columns={target_col:"sub", tx:"x", ty:"y"}),
            in_edges[[source_col, sx, sy]].rename(columns={source_col:"sub", sx:"x", sy:"y"}),
            in_edges[[target_col, tx, ty]].rename(columns={target_col:"sub", tx:"x", ty:"y"}),
        ], ignore_index=True).drop_duplicates(subset=["sub"])

        center = nodes[nodes["sub"] == subreddit_name]
        others = nodes[nodes["sub"] != subreddit_name]

        frame_data = [
            # outgoing edges
            go.Scatter(
                x=out_xs, y=out_ys, mode="lines",
                line=dict(width=1), opacity=0.7,
                name="Outgoing", hoverinfo="skip"
            ),
            go.Scatter(
                x=others["x"], y=others["y"],
                mode="markers+text" if show_labels else "markers",
                text=others["sub"] if show_labels else None,
                textposition="top center",
                marker=dict(size=7),
                hovertext=others["sub"],
                hoverinfo="text",
                name="Other subreddits"
            ),
            # center node
            go.Scatter(
                x=center["x"] if not center.empty else [np.nan],
                y=center["y"] if not center.empty else [np.nan],
                mode="markers+text" if show_labels else "markers",
                text=[subreddit_name] if (show_labels and not center.empty) else None,
                textposition="top center",
                marker=dict(size=12, symbol="star"),
                hovertext=[subreddit_name] if not center.empty else None,
                hoverinfo="text",
                name=f"r/{subreddit_name}"
            ),
        ]

        frames.append(go.Frame(name=str(p.date()), data=frame_data))

    # Initial figure uses first frame
    fig = go.Figure(data=frames[0].data, frames=frames)

    # Slider + play/pause
    steps = [
        dict(
            method="animate",
            args=[[f.name], {"mode": "immediate", "frame": {"duration": 400, "redraw": True}, "transition": {"duration": 200}}],
            label=f.name
        )
        for f in frames
    ]

    fig.update_layout(
        template=template,
        title=f"2D Connection Map Over Time — r/{subreddit_name} ({freq})",
        xaxis=dict(title="x", range=[x_min, x_max]),
        yaxis=dict(title="y", range=[y_min, y_max]),
        legend_title="",
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.1, y=1.15,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 500, "redraw": True}, "transition": {"duration": 200}, "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0,
            x=0.1, y=1.05,
            len=0.8,
            steps=steps
        )]
    )

    fig.show()


def plot_top_links_with_subreddit_time_slider(
    df,
    subreddit_name,
    source_col="SOURCE_SUBREDDIT",
    target_col="TARGET_SUBREDDIT",
    sentiment_col="LINK_SENTIMENT",
    negative_value=-1,
    time_col="TIMESTAMP",
    freq="M"  # "D", "W", or "M"
):
    df = df.copy()

    # Keep only NEGATIVE links
    df = df[df[sentiment_col] == negative_value]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Time binning
    df["period"] = df[time_col].dt.to_period(freq).dt.to_timestamp()

    # Keep only interactions involving the subreddit
    mask = (df[source_col] == subreddit_name) | (df[target_col] == subreddit_name)
    df_sub = df[mask].copy()

    # Define the "other" subreddit
    df_sub["other_sub"] = df_sub.apply(
        lambda r: r[target_col] if r[source_col] == subreddit_name else r[source_col],
        axis=1
    )

    # Aggregate counts per period
    agg = (
        df_sub
        .groupby(["period", "other_sub"])
        .size()
        .reset_index(name="n_links")
    )

    # Keep Top-10 per period
    agg["rank"] = agg.groupby("period")["n_links"].rank(method="first", ascending=False)
    top10 = agg[agg["rank"] <= 10]

    # --- global fixed x-range ---
    global_max = top10["n_links"].max()
    if pd.isna(global_max) or global_max <= 0:
        global_max = 1
    pad = 0.05 * global_max
    x_range = [0, global_max + pad]

    # Plot
    fig = px.bar(
        top10,
        x="n_links",
        y="other_sub",
        color="n_links",
        orientation="h",
        animation_frame="period",
        title=f"Top 10 Subreddits Most Negatively Linked With r/{subreddit_name}",
        template="plotly_dark"
    )

    # Force fixed x-axis on main layout
    fig.update_xaxes(autorange=False, range=x_range)

    # 🔥 Force fixed x-axis on EVERY frame (this is the key)
    for fr in fig.frames:
        fr.layout = go.Layout(xaxis=dict(autorange=False, range=x_range))

    fig.update_layout(
        xaxis_title="Number of negative links",
        yaxis_title="Subreddit",
        yaxis={"categoryorder": "total ascending"},
        transition={"duration": 300}
    )

    fig.show()
