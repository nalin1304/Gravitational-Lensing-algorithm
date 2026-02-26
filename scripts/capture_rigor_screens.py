import asyncio
from playwright.async_api import async_playwright
import os

ARTIFACT_DIR = "/Users/nalinaggarwal/.gemini/antigravity/brain/b8ba6afc-cc39-4c77-b3a2-bb60a516248e"

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Go to rigor page
        await page.goto("http://127.0.0.1:8000/ui#/rigor")
        await page.wait_for_selector("#rigorTabs")
        
        # wait a second for rendering
        await asyncio.sleep(1)
        
        # 1. SBI NPE
        await page.click("button[data-rtab='sbi']")
        await page.click("#sbiRunBtn")
        await page.wait_for_selector("#sbiResults .metric-val")
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "rigor_sbi_npe.png"))
        
        # 2. Starlets
        await page.click("button[data-rtab='starlet']")
        await page.click("#starletRunBtn")
        await page.wait_for_selector("#starletResults img")
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "rigor_starlets.png"))
        
        # 3. SED
        await page.click("button[data-rtab='sed']")
        await page.click("#sedRunBtn")
        await page.wait_for_selector("#sedResults .metric-val")
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "rigor_sed.png"))
        
        # 4. Env Linker
        await page.click("button[data-rtab='env']")
        await page.click("#envRunBtn")
        await page.wait_for_selector("#envResults .metric-val")
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "rigor_env.png"))
        
        # 5. Gate Validator
        await page.click("button[data-rtab='gate']")
        await page.click("#gateRunBtn")
        await page.wait_for_selector("#gateResults .metric-val")
        await page.screenshot(path=os.path.join(ARTIFACT_DIR, "rigor_gate.png"))
        
        await browser.close()
        print("Screenshots captured successfully.")

if __name__ == "__main__":
    asyncio.run(run())
