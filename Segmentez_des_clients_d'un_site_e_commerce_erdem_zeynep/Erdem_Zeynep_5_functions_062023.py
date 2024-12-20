#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import missingno as msno
import sklearn
import warnings


def pourcentage_null_text(df):
    """
    On donne l'infos pourcentage des valeurs nulles pour le dataframe en texte
    :param df: dataframe
    :return: none
    """
    all_num_data = df.shape[0] * df.shape[1]
    num_null = df.isnull().sum().sum()
    pourcentage_null = ((num_null / all_num_data) * 100).round(2)
    print("** On a ", pourcentage_null, "% de valeur nulle dans notre dataframe **")
    return pourcentage_null


def per_null_dataframe(df):
    """
    On donne l'infos pourcentage des valeurs nulles pour le dataframe en détaille
    On display pourcentage des valeurs nulles par colonne en descendant
    :param df: dataframe
    :return: le dataframe des pourcentages
    """
    per_null = (df.isnull().mean() * 100).round(2)
    nombre_null = df.isnull().sum()
    null_df = pd.concat([nombre_null, per_null], axis=1)
    null_df.columns = ["nombre_null", "pourcentage_null"]
    return null_df.sort_values("nombre_null", ascending=False)


def info_general(df):
    """
    On donne les infos générales pour le dataframe
    :param df: dataframe
    :return: none
    """
    print("*" * 25, "* INFORMATIONS GENERALES DE NOTRE DATAFRAME *", "*" * 25)
    print("-" * 100)
    shape_df = df.shape
    print("" * 100)
    print(
        "Il y a",
        shape_df[0],
        "lignes et",
        shape_df[1],
        "colonnes dans notre dataframe.",
    )
    print("-" * 100, sep="\n")
    print("* On obtiens l'info sur notre dataframe *", "" * 100)
    print("-" * 100, sep="\n")
    print(df.info())
    print("-" * 100, sep="\n")
    print("* On obtient les informations statistiques sur notre dataframe *", "" * 100)
    print("-" * 100, sep="\n")
    display(df.describe(include="all").T)
    print("-" * 100, sep="\n")
    print("*On vérifie s'il y a des doublons dans notre dataframe*", "" * 100)
    search_dup = df.duplicated().sum()
    if search_dup == 0:
        print("** Il n'y a pas de doublons **")
    else:
        print("** Il y a ", search_dup, " doublons **")
    print("-" * 100)
    print(
        "** On vérifie s'il y a des valeurs nulles et "
        "on display pourcentage des valeurs nulles par colonne en descendant **"
    )
    print("-" * 100)
    pourcentage_null = pourcentage_null_text(df)
    if pourcentage_null > 0:
        display(per_null_dataframe(df))
        print("** Visualisation des valeurs nulles **", "" * 100)
        msno.bar(df)
    else:
        pourcentage_null
    print("-" * 100, sep="\n")

    return


def display_nulls(df, col):
    """
    On display des valeurs nulles pour le dataframe
    :param df: dataframe
    :param col: la colonne qu'on cherche les valeurs nulles
    :return: le dataframe avec des valeurs nulles
    """
    display_null = pd.isnull(df[col])
    df_null = df[display_null]
    return df_null


def delete_nulls(df, col):
    """
    On supprime des valeurs nulles pour le dataframe
    :param df: dataframe
    :param col: la colonne qu'on va supprimer les valeurs nulles
    :return: le dataframe avec des valeurs sans nulles
    """
    display_null = pd.isnull(df[col])
    df_sans_null = df[~display_null]
    return df_sans_null


def detect_outliers(df, col, ecart_accepte):
    """
    On trouve les valeurs abérrantes dans la dataframe
    :param df: dataframe
    :col: la colonne de dataframe
    :return: les limits des outliers et les valeurs des outliers
    """
    q1 = df[col].quantile(q=0.25)
    q3 = df[col].quantile(q=0.75)
    diff = q3 - q1
    lower_bound = q1 - ecart_accepte * diff
    upper_bound = q3 + ecart_accepte * diff
    outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
    print("lower_bound =", lower_bound, "upper_bound =", upper_bound)
    return outliers


def delete_all_nulls(df):
    """
    On supprime des colonnes complètement nulles pour le dataframe
    :param df: dataframe
    :return: le dataframe avec des valeurs sans nulles
    """
    null_df = (df.isnull().mean() * 100).round(2).reset_index()
    null_df.columns = ["colonne", "pourcentage_null"]
    null_df_full = null_df[null_df["pourcentage_null"] == 100]
    col_not_null = null_df[null_df["pourcentage_null"] < 100]["colonne"]
    df_col_notnull = df[col_not_null]
    print("On a supprimé", null_df_full.shape[0], " colonnes.")
    return df_col_notnull


def heatmap(data):
    sns.set(font_scale=1.2)
    fig = plt.figure(figsize=(20, 8))
    matrix = data.corr()
    plt.title("Le Heatmap de la Data", size=25)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", mask=mask)
    plt.xticks(rotation=90)
    return


def impute_missing_data(df):
    X = pd.concat(
        [pd.get_dummies(df.select_dtypes("O")), df.select_dtypes(exclude="O")], axis=1
    )

    imputer = KNNImputer(n_neighbors=4)
    imputation_res = pd.DataFrame(imputer.fit_transform(X))
    imputation_res.columns = X.columns
    imputation_res = imputation_res[
        [
            col
            for col in imputation_res.columns
            if not col.startswith("PrimaryPropertyType")
        ]
    ]
    imputation_res = imputation_res.reset_index(drop=True)
    imputation_res.insert(0, "PrimaryPropertyType", df["PrimaryPropertyType"])
    return imputation_res


def drop_duplicates(df, column):
    # Find duplicate rows based on the specified column
    duplicate_rows = df.duplicated(subset=column)

    if duplicate_rows.any():
        # Count the number of duplicate rows
        num_duplicates = duplicate_rows.sum()

        # Drop the duplicate rows based on the specified column
        df.drop_duplicates(subset=column, inplace=True)
        print(f"{num_duplicates} duplicate rows have been dropped.")
    else:
        print("No duplicate rows found.")
    return df


def countplot_top10(df, col, figsize=(9, 4)):
    unique_values = df[col].nunique()

    df_ranked = df[col].astype(str).value_counts().reset_index()
    df_ranked.columns = [col, "Count"]
    df_ranked = df_ranked.sort_values("Count", ascending=False).head(10)

    total_count = df_ranked["Count"].sum()
    df_ranked["Percentage"] = (df_ranked["Count"] / total_count) * 100
    print("Il y a ", unique_values, "unique ", col, "dans notre data.")
    sns.set_style("white")  # Remove gridlines
    plt.figure(figsize=(figsize[0], figsize[1]))  # Adjust the figure size as needed

    ax = sns.barplot(x="Count", y=col, data=df_ranked, orient="h")
    plt.xlabel("Count", size=11)
    plt.ylabel(col, size=13)
    plt.title("Top 10 Répartition du " + col, size=13)

    for i in range(df_ranked.shape[0]):
        count = df_ranked["Count"].iloc[i]
        percentage = df_ranked["Percentage"].iloc[i]
        ax.text(count, i, f"{percentage:.2f}%", va="center", fontdict=dict(fontsize=10))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    # Adjust the x-axis limits if needed
    plt.xlim(0, df_ranked["Count"].max() * 1.1)

    plt.tight_layout()
    plt.show()


def countplot(df, col, figsize=(9, 4)):
    df_ranked = df[col].astype(str).value_counts().reset_index()
    df_ranked.columns = [col, "Count"]
    df_ranked = df_ranked.sort_values("Count", ascending=False)

    total_count = df_ranked["Count"].sum()
    df_ranked["Percentage"] = (df_ranked["Count"] / total_count) * 100

    sns.set_style("white")  # Remove gridlines
    plt.figure(figsize=(figsize[0], figsize[1]))  # Adjust the figure size as needed

    ax = sns.barplot(x="Count", y=col, data=df_ranked, orient="h")
    plt.xlabel("Count", size=11)
    plt.ylabel(col, size=13)
    plt.title("Répartition du " + col, size=13)

    for i in range(df_ranked.shape[0]):
        count = df_ranked["Count"].iloc[i]
        percentage = df_ranked["Percentage"].iloc[i]
        ax.text(count, i, f"{percentage:.2f}%", va="center", fontdict=dict(fontsize=10))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    # Adjust the x-axis limits if needed
    plt.xlim(0, df_ranked["Count"].max() * 1.1)

    plt.tight_layout()
    plt.show()


def distplot_and_boxplot(df, columns, figsize=(12, 3)):
    if not isinstance(columns, list):
        columns = [columns]

    num_columns = len(columns)
    fig, axes = plt.subplots(
        num_columns, 2, figsize=(figsize[0], figsize[1] * num_columns)
    )

    for i, col in enumerate(columns):
        if num_columns == 1:  # Adjust subplots if only one column is provided
            ax_distplot = axes[0]
            ax_boxplot = axes[1]
        else:
            ax_distplot = axes[i, 0]
            ax_boxplot = axes[i, 1]

        # Distplot
        sns.histplot(df[col], ax=ax_distplot, kde=True)
        ax_distplot.set_title("Distribution Plot - " + col)
        ax_distplot.set_xlabel(col)

        # Boxplot
        sns.boxplot(x=df[col], ax=ax_boxplot)
        ax_boxplot.set_title("Box Plot - " + col)
        ax_boxplot.set_xlabel(col)

    plt.tight_layout()
    plt.show()


def display_circles(
    pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None
):
    """
    Trace les cercles de corrélations
    """
    for (
        d1,
        d2,
    ) in (
        axis_ranks
    ):  # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(5, 5))

            # détermination des limites du graphique
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else:
                xmin, xmax, ymin, ymax = (
                    min(pcs[d1, :]),
                    max(pcs[d1, :]),
                    min(pcs[d2, :]),
                    max(pcs[d2, :]),
                )

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(
                    np.zeros(pcs.shape[1]),
                    np.zeros(pcs.shape[1]),
                    pcs[d1, :],
                    pcs[d2, :],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="grey",
                )
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(
                    LineCollection(lines, axes=ax, alpha=0.1, color="black")
                )

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(
                            x,
                            y,
                            labels[i],
                            fontsize="14",
                            ha="center",
                            va="center",
                            rotation=label_rotation,
                            color="black",
                            alpha=0.5,
                        )

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="b")
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color="grey", ls="--")
            plt.plot([0, 0], [-1, 1], color="grey", ls="--")

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel(
                "F{} ({}%)".format(
                    d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)
                )
            )
            plt.ylabel(
                "F{} ({}%)".format(
                    d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)
                )
            )

            plt.axis("square")
            plt.grid(True)
            plt.title("Cercle des corrélations (F{} et F{})".format(d1 + 1, d2 + 1))
            plt.show(block=False)
            plt.show()


def display_factorial_planes(
    X_projected,
    n_comp,
    pca,
    axis_ranks,
    couleurs=None,
    labels=None,
    n_cols=3,
    alpha=1,
    illustrative_var=None,
    lab_on=True,
    size=10,
):
    for i, (d1, d2) in enumerate(axis_ranks):
        if d2 < n_comp:
            # initialisation de la figure
            fig = plt.figure(figsize=(5, 5))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha, s=size)
            else:
                illustrative_var = np.array(illustrative_var)
                label_patches = []
                colors = couleurs
                i = 0

                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(
                        X_projected[selected, d1],
                        X_projected[selected, d2],
                        alpha=alpha,
                        label=value,
                        c=colors[i],
                    )
                    label_patch = mpatches.Patch(color=colors[i], label=value)
                    label_patches.append(label_patch)
                    i += 1
                    ax.legend(
                        handles=label_patches,
                        bbox_to_anchor=(1.05, 1),
                        loc=2,
                        borderaxespad=0.0,
                        facecolor="white",
                    )
                plt.legend()

            # affichage des labels des points
            if labels is not None and lab_on:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i], fontsize="14", ha="center", va="center")

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color="grey", ls="--")
            plt.plot([0, 0], [-100, 100], color="grey", ls="--")

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel(
                "F{} ({}%)".format(
                    d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)
                )
            )
            plt.ylabel(
                "F{} ({}%)".format(
                    d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)
                )
            )

            plt.title(
                "Projection des individus (sur F{} et F{})".format(d1 + 1, d2 + 1)
            )
            plt.show(block=False)
            # plt.grid(False)


def display_scree_plot(pca):
    taux_var_exp = pca.explained_variance_ratio_
    scree = pca.explained_variance_ratio_ * 100
    plt.bar(np.arange(len(scree)) + 1, scree)
    ax1 = plt.gca()
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker="o")
    plt.axhline(y=90, color="r")
    plt.text(2, 92, ">90%", color="r", fontsize=10)
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    for i, p in enumerate(ax1.patches):
        ax1.text(
            p.get_width() / 5 + p.get_x(),
            p.get_height() + p.get_y() + 0.3,
            "{:.0f}%".format(taux_var_exp[i] * 100),
            fontsize=8,
            color="k",
        )
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
