#!/usr/bin/env python
# coding: utf-8

"""
10-K form parser module.

Functions:
    parse_10k_filing(file_path, section): Parses a SEC 10-K filing and extracts sections.
    parse_all_forms(df, form, header): Iterates over a filings DataFrame and creates a result DataFrame.
"""

import re
import sys
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup as bs
import logging
from progiter.manager import ProgressManager  # added import

logger = logging.getLogger(__name__)

def parse_10k_filing(file_path, section):
    """
    Parse the 10-K filing at file_path and extract requested sections.
    
    Args:
        file_path (str): Path to the filing file.
        section (int): Section option [0(All), 1(Business), 2(Risk), 3(MDA)].
    
    Returns:
        list: Extracted text sections.
    """
    if section not in [0, 1, 2, 3]:
        logger.error("Not a valid section")
        sys.exit(1)
    
    def get_text(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        html = bs(content, 'html.parser')
        text = html.get_text()
        text = unicodedata.normalize("NFKD", text).encode('ascii', 'ignore').decode('utf8')
        return " ".join(text.split("\n"))
    
    def extract_text(text, item_start, item_end):
        starts = [m.start() for m in item_start.finditer(text)]
        ends = [m.start() for m in item_end.finditer(text)]
        positions = []
        for s in starts:
            for e in ends:
                if s < e:
                    positions.append([s, e])
                    break  # take first valid match per start
        if positions:
            item_position = max(positions, key=lambda p: p[1]-p[0])
            return text[item_position[0]:item_position[1]]
        return ""
    
    text = get_text(file_path)
    
    try:
        if section in [1, 0]:
            item1_start = re.compile(r"item\s*[1][\.\;\:\-\_]*\s*\b", re.IGNORECASE)
            item1_end = re.compile(r"item\s*1a[\.\;\:\-\_]\s*Risk|item\s*2[\.,\;\:\-\_]\s*Prop", re.IGNORECASE)
            businessText = extract_text(text, item1_start, item1_end)
        if section in [2, 0]:
            item1a_start = re.compile(r"(?<!,\s)item\s*1a[\.\;\:\-\_]\s*Risk", re.IGNORECASE)
            item1a_end = re.compile(r"item\s*2[\.\;\:\-\_]\s*Prop|item\s*[1][\.\;\:\-\_]*\s*\b", re.IGNORECASE)
            riskText = extract_text(text, item1a_start, item1a_end)
        if section in [3, 0]:
            item7_start = re.compile(r"item\s*[7][\.\;\:\-\_]*\s*\bM", re.IGNORECASE)
            item7_end = re.compile(r"item\s*7a[\.\;\:\-\_]\s+Quanti|item\s*8[\.,\;\:\-\_]\s*", re.IGNORECASE)
            mdaText = extract_text(text, item7_start, item7_end)
    except Exception as ex:
        logger.error("Error during parsing: %s", ex)
        return ["Error"]
    
    if section == 0:
        return [businessText, riskText, mdaText]
    elif section == 1:
        return [businessText]
    elif section == 2:
        return [riskText]
    elif section == 3:
        return [mdaText]

def parse_all_forms(df, form, header):
    """
    Parse filings from a DataFrame and return a consolidated DataFrame with parsed texts.
    
    Args:
        df (pd.DataFrame): DataFrame containing filings metadata.
        form (str): Filing form to filter.
        header (dict): HTTP headers if needed (unused in this parser).
    
    Returns:
        pd.DataFrame: DataFrame with extracted text and metadata.
    """
    df_filtered = df[df.form == form]
    results = pd.DataFrame()
    with ProgressManager() as pman:
        pman = pman or ProgressManager()  # fallback if None
        for _, row in pman.progiter(df_filtered.iterrows(), total=df_filtered.shape[0], desc="Parsing filings"):
            file_path = f"data/{row['ticker']}/{row['primaryDocument']}"
            section = 0  # To extract all sections
            text_data = parse_10k_filing(file_path, section)
            temp_df = pd.DataFrame({'Text': text_data})
            temp_df['ticker'] = row['ticker']
            temp_df['accessionNumber'] = row['accessionNumber']
            temp_df['cik'] = row['cik']
            temp_df['filepath'] = file_path
            results = pd.concat([results, temp_df], ignore_index=True)
    return results
