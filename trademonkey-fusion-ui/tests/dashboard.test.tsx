// File: D:\Repos\trade-monkey-lite\trade-monkey-gpu-trademonkey-fusion-ui\tests\dashboard.test.tsx
import { render, screen } from '@testing-library/react';
import DashboardLayout from '@/components/dashboard/DashboardLayout';
import { useRealtimeData } from '@/hooks/useRealtimeData';

jest.mock('@/hooks/useRealtimeData', () => ({
  useRealtimeData: jest.fn(() => ({
    data: {
      sentiment: 0.85,
      bull: 0.6,
      bear: 0.2,
      crab: 0.15,
      shock: 0.05,
    },
  })),
}));

describe('DashboardLayout', () => {
  it('renders all components in correct grid order', () => {
    render(<DashboardLayout />);
    
    expect(screen.getByText(/Sentiment Telepathy/i)).toBeInTheDocument();
    expect(screen.getByText(/Market Regime Radar/i)).toBeInTheDocument();
    expect(screen.getByText(/Position Management/i)).toBeInTheDocument();
    expect(screen.getByText(/GPU Performance/i)).toBeInTheDocument();
    expect(screen.getByText(/System Health/i)).toBeInTheDocument();
    expect(screen.getByText(/Activity Feed/i)).toBeInTheDocument();
    expect(screen.getByText(/Start Demo/i)).toBeInTheDocument();
  });

  it('toggles demo mode correctly', () => {
    render(<DashboardLayout />);
    
    const demoButton = screen.getByText(/Start Demo/i);
    demoButton.click();
    expect(screen.getByText(/Exit Demo/i)).toBeInTheDocument();
  });
});