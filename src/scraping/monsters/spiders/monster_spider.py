import scrapy
from ..items import MonsterItem
import time
from functools import reduce


query = 'bestiary/'


class MonsterSpider(scrapy.Spider):
    name = 'monster_spider'
    allowed_domains = ['chisaipete.github.io']
    start_urls = ['http://chisaipete.github.io/' + query]


    def parse(self, response):

        names = response.xpath('//a[@class="post-link"]/text()').getall()
        links = [response.url + 'creature/' + str(i).lower() for i in names]
        links = [url.replace(' ', '-').replace('(', '').replace(')','') for url in links]
        for url in links:
            yield scrapy.Request(url, callback=self.parse_statblock)

    
    def parse_statblock(self, response):
        item = MonsterItem()

        item['name'] = response.xpath('//div[@class="creature-heading"]/h1/text()').get()
        item['short_desc'] = response.xpath('//div[@class="creature-heading"]/h2/text()').get()
        item['armor_class'] = response.xpath('//div[@class="property-line first"]/p/text()').get()
        item['speed'] = response.xpath('//div[@class="property-line last"]/p/text()').get()
        item['strength'] = response.xpath('//div[@class="ability-strength"]/p/text()').get()
        item['dexterity'] = response.xpath('//div[@class="ability-dexterity"]/p/text()').get()
        item['constitution'] = response.xpath('//div[@class="ability-constitution"]/p/text()').get()
        item['intelligence'] = response.xpath('//div[@class="ability-intelligence"]/p/text()').get()
        item['wisdom'] = response.xpath('//div[@class="ability-wisdom"]/p/text()').get()
        item['charisma'] = response.xpath('//div[@class="ability-charisma"]/p/text()').get()
        item['url'] = response.url
        
        # Parsing the actions block of each stat block
        actions_count = len(response.xpath('//div[@class="section-left"]/p').getall())
        actions_list = []
        for p_tag in range(actions_count):
            p_tag += 1
            append_string = ''
            ability_name = response.xpath(f'//div[@class="section-left"]/p[{p_tag}]/strong/em[1]/text()').get()
            if ability_name:
                append_string += "paragraph="
                append_string += str(p_tag)
                append_string += ability_name
                append_string += ";;"
            else:
                append_string += "paragraph="
                append_string += str(p_tag)

            append_string += response.xpath(f'//div[@class="section-left"]/p[{p_tag}]/text()').get()

            if ability_name == "Spellcasting.":
                spell_list = response.xpath('//div[@class="section-left"]/ul/li/p/text()').getall()
                spells = '++'.join(spell_list)
                append_string += spells
                
            actions_list.append(append_string)

        try:   
            for num, ability in enumerate(actions_list):
                item[f'ability_{num + 1}'] = ability
        except:
            item['misc'] = reduce(lambda x, y: x + y, actions_list)

        # Parsing the specifics block of each stat block
        specs_heads = response.xpath('//div[@class="property-line"]/h4/text()').getall()
        specs = response.xpath('//div[@class="property-line"]/p/text()').getall()
        mesh = zip(specs_heads, specs)

        for head, element in mesh:
            if head == 'Hit Points':
                item['hit_points'] = element
            elif head == 'Damage Immunities':
                item['damage_immunities'] = element
            elif head == 'Damage Resistances':
                item['damage_resistances'] = element
            elif head == 'Condition Immunities':
                item['condition_immunities'] = element
            elif head == 'Senses':
                item['senses'] = element
            elif head == 'Languages':
                item['languages'] = element
            elif head == 'Challenge':
                item['challenge'] = element
            elif head == 'Saving Throws':
                item['saving_throws'] = element
            elif head == 'Skills':
                item['skills'] = element
        
        yield item

