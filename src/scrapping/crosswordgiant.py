import requests
from typing import List, Dict
from bs4 import BeautifulSoup
import csv


def sanitize(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return "".join(x for x in name if x.isalnum() or x == "_")


# Assuming the HTML content provided is saved in a variable called 'html_data'
# or read from a file.
def parse_crossword_solutions(html_content: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_content, "html.parser")

    # The clues and solutions are located within <tr> tags with class "gradeX"
    rows = soup.find_all("tr", class_="gradeX")

    clue_answer_pairs = []

    for row in rows:
        # Each row has two <td> (table data) cells
        cells = row.find_all("td")

        if len(cells) >= 2:
            # The first cell contains the Definition (Clue)
            # The second cell contains the Solution (Answer)
            clue = cells[0].get_text(strip=True)
            answer = cells[1].get_text(strip=True)

            clue_answer_pairs.append({"clue": clue, "answer": answer})

    return clue_answer_pairs


def extract_grid_links(html_content) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_content, "html.parser")

    # Grid links are inside <tr> tags with class "gradeX"
    rows = soup.find_all("tr", class_="gradeX")

    grid_links = []

    for row in rows:
        link_tag = row.find("a")
        if link_tag:
            href = link_tag.get("href", "")

            # Filter to ensure we only get the puzzle links,
            # as sometimes rows might contain other links.
            if "/crossword-puzzle/" in href:
                # Handle protocol-relative URLs (e.g., //www.example.com)
                full_url = "https:" + href if href.startswith("//") else href

                grid_links.append(
                    {"name": link_tag.get_text(strip=True), "url": full_url}
                )

    return grid_links


def scrap_publisher(base_url: str) -> Dict[str, List[Dict[str, str]]]:
    k = 1
    releases = {}
    print(base_url)
    while True:
        print(k)
        try:
            r = requests.get(base_url + f"?page={k}")
            if r.status_code == 200:
                grid_links = extract_grid_links(r.text)
                if len(grid_links) == 0:
                    break
                for link in grid_links:
                    r = requests.get(link["url"])
                    releases[link["name"]] = parse_crossword_solutions(r.text)
                k += 1
            else:
                break
        except Exception as e:
            print(e)
            continue
    return releases


def process_all(base_urls: List[str], output_path: str):
    for base_url in base_urls:
        releases = scrap_publisher(base_url)
        for name, clue_answer_pairs in releases.items():
            filename = output_path + "/" + sanitize(name) + ".csv"
            with open(filename, "w", encoding="utf8") as f:
                writer = csv.DictWriter(f, fieldnames=["clue", "answer"])
                writer.writeheader()
                for clue_answer_pair in clue_answer_pairs:
                    writer.writerow(clue_answer_pair)


if __name__ == "__main__":
    process_all(
        [
            "https://www.crosswordgiant.com/fr/browse/36/La-Presse",
            "https://www.crosswordgiant.com/fr/browse/38/Le-Devoir",
        ],
        "data/list_clue_answers",
    )
