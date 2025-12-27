# 🚀 Deployment Guide

This guide will help you deploy your **Movie Recommendation System** to **Streamlit Community Cloud**. It is the easiest and best way to host Streamlit apps for free.

## 1. Prerequisites (GitHub)

1.  **Create a GitHub Account** (if you don't have one): [https://github.com/signup](https://github.com/signup)
2.  **Initialize a Repository**:
    *   Since you have the code on your computer, you need to push it to GitHub.
    *   If you have GitHub Desktop or use the command line:
        ```bash
        git init
        git add .
        git commit -m "Initial commit"
        # Create a new repo on GitHub.com called 'movie-recommender'
        # Then link it:
        git remote add origin https://github.com/YOUR_USERNAME/movie-recommender.git
        git push -u origin master
        ```
    *   **Note**: We have successfully configured `.gitignore` to **ignore** the large `movie_data.pkl` file (190MB). GitHub has a 100MB file limit, so this is critical. The app is smart enough to rebuild the model from the CSV files when it starts!

## 2. Deploy to Streamlit Cloud

1.  Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2.  Click **"New app"**.
3.  Select your repository (`movie-recommender`), branch (`master` or `main`), and the main file path (`app.py`).
4.  Click **"Deploy!"**.

## 3. Deployment Notes

*   **First Run**: The first time the app loads, it will take about **20-30 seconds** to build the model (processing the 5000 movies). You will see a "Building recommendation model..." spinner.
*   **Subsequent Runs**: Streamlit will cache the data in memory, so it will be fast after that.
*   **API Keys**: The current code uses a hardcoded TMDB API key. For a production app, it is safer to use "Secrets", but for this personal project, the current setup will work fine.

## Troubleshooting

*   **"FileNotFoundError"**: Ensure `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` (or the .zip) are in the GitHub repository.
*   **Memory Limit**: If the app crashes with an "Out of Memory" error, Streamlit Cloud's free tier might be too small for the 5000 movie matrix.
    *   *Fix*: Reduce `max_features` in `app.py` from `5000` to `3000` in the `TfidfVectorizer` line.

---
**Enjoy your Movie Recommender! 🍿**
