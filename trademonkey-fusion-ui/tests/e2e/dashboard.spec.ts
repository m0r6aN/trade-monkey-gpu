// File: D:\Repos\trade-monkey-lite\trade-monkey-gpu-trademonkey-fusion-ui\tests\e2e\dashboard.spec.ts
import { test, expect, Page } from '@playwright/test';

test.describe('Dashboard E2E', () => {
  test('Bull market scenario displays correctly', async ({ page }: { page: Page }) => {
    await page.goto('/dashboard');
    
    await page.click('text=Start Demo');
    await page.selectOption('select#scenario', 'bull');
    
    await expect(page.locator('.sentiment-gauge')).toHaveCSS('background-color', /hsl\(142, 85%, 50%\)/);
    await expect(page.locator('.signal-boost')).toContainText('+35%');
    await expect(page.locator('.market-regime')).toContainText('Bull 60%');
  });

  test('Mobile navigation works', async ({ page }: { page: Page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/dashboard');
    
    await page.click('.mobile-menu-toggle');
    await page.click('text=Positions');
    
    await expect(page).toHaveURL(/positions/);
    await expect(page.locator('.position-panel')).toBeVisible();
  });
});
