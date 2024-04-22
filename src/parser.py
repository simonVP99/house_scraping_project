from bs4 import BeautifulSoup

def parse_page(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    houses = []
    for element in soup.find_all('div', class_='house-info'):
        house_data = {
            'title': element.find('h2').text,
            'price': element.find('span', class_='price').text,
            'details': element.find('p', class_='details').text
        }
        houses.append(house_data)
    return houses
