from django.shortcuts import render


def index(request):
    students = [
        {"name": "Praveen Megharaj", "matriculation": "672067"},
        {"name": "Niharika Kiran",   "matriculation": "672070"},
    ]
    projects = [
        {"name": "Project 1", "title": "Supervised Learning",
         "blurb": "Upload a tabular dataset, explore it, and train a classical model with an interactive hyperparameter sweep.",
         "url_name": "project1:index"},
        {"name": "Project 2", "title": "Explainability",
         "blurb": "Palmer Penguins with complexity-controlled models, counterfactuals, and PDP + ALE feature effects.",
         "url_name": "project2:index"},
        {"name": "Project 3", "title": "Learning to Defer",
         "blurb": "AG News classification with a simulated expert, a deferral policy, and active learning for expert competence.",
         "url_name": "project3:index"},
        {"name": "Project 4", "title": "Preference Elicitation",
         "blurb": "Movie-recommender user study comparing pairwise choice vs ranking-of-ten with a Plackett-Luce preference model.",
         "url_name": "project4:index"},
    ]
    return render(request, "home/index.html",
                  {"students": students, "projects": projects})
