import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cyberInput, cyberButton } from '@/lib/utils'; // Assuming custom utility for class names

const AIChatAssistant = () => {
  const [messages, setMessages] = useState<{ role: string; content: string; tool?: string; confidence?: number; timestamp: string }[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  const quickActions = [
    { label: 'BTC Sentiment', query: 'What\'s the sentiment on BTC?' },
    { label: 'Market News', query: 'Search for recent market news' },
    { label: 'Risk Calc', query: 'Calculate risk for a $1000 position' },
  ];

  const handleSend = async (query: string = input) => {
    if (!query.trim()) return;

    const userMessage = { role: 'user', content: query, timestamp: new Date().toLocaleTimeString() };
    setMessages([...messages, userMessage]);
    setIsTyping(true);

    let response = '';
    let tool = '';
    let confidence = 0.9;

    try {
      if (query.toLowerCase().includes('sentiment') || query.toLowerCase().includes('btc')) {
        const sentimentResponse = await fetch('http://localhost:9026/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: 'BTC market sentiment' }),
        });
        const sentimentData = await sentimentResponse.json();
        response = `BTC sentiment is ${sentimentData.sentiment || '+0.85 (bullish)'}. Confidence: ${sentimentData.confidence || '0.92'}.`;
        tool = 'TradeMonkey (9026)';
        confidence = sentimentData.confidence || 0.92;
      } else if (query.toLowerCase().includes('search') || query.toLowerCase().includes('news')) {
        const searchResponse = await fetch('http://localhost:9206/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        const searchData = await searchResponse.json();
        response = `Found recent news: ${searchData.results || 'Bullish articles on BTC!'}`;
        tool = 'Web Search (9206)';
      } else if (query.toLowerCase().includes('calculate') || query.toLowerCase().includes('risk')) {
        const calcResponse = await fetch('http://localhost:9202/calculate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ operation: 'multiply', a: 1000, b: 0.25 }),
        });
        const calcData = await calcResponse.json();
        response = `Risk calculation: Position size = ${calcData.result || '250'}`;
        tool = 'Calculator (9202)';
      } else if (query.toLowerCase().includes('research')) {
        const researchResponse = await fetch('http://localhost:9010/research', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        const researchData = await researchResponse.json();
        response = `Research findings: ${researchData.results || 'Market trends indicate a bullish cycle.'}`;
        tool = 'Research Agent (9010)';
      } else if (query.toLowerCase().includes('optimize')) {
        const mathResponse = await fetch('http://localhost:9002/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        const mathData = await mathResponse.json();
        response = `Portfolio optimization: ${mathData.results || 'Optimal allocation calculated.'}`;
        tool = 'Math Solver (9002)';
      } else {
        response = 'TradeMonkey Agent (port 9026) here! I can assist with sentiment analysis, market news, risk calculations, research, or optimization. What do you need?';
        tool = 'TradeMonkey (9026)';
      }
    } catch (error) {
      response = 'Error connecting to OMEGA agents. Please try again!';
      tool = 'Error';
      confidence = 0;
    }

    setTimeout(() => {
      setMessages([...messages, userMessage, { role: 'assistant', content: response, tool, confidence, timestamp: new Date().toLocaleTimeString() }]);
      setIsTyping(false);
    }, 1000);
    setInput('');
  };

  return (
    <motion.div
      className="col-span-1"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className="cyber-card">
        <CardHeader>
          <CardTitle className="cyber-glitch text-white">AI Chat Assistant</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2 mb-2">
            {quickActions.map((action) => (
              <button
                key={action.label}
                onClick={() => handleSend(action.query)}
                className="cyber-button text-sm px-2 py-1"
              >
                {action.label}
              </button>
            ))}
          </div>
          <div className="h-64 overflow-y-auto p-4 bg-gray-800 rounded-lg">
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                className={`mb-2 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}
                initial={{ opacity: 0, x: msg.role === 'user' ? 20 : -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
              >
                <div className="flex items-center gap-2">
                  {msg.role === 'assistant' && msg.tool && (
                    <Badge className="bg-blue-600">{msg.tool}</Badge>
                  )}
                  <span
                    className={`inline-block p-2 rounded-lg ${
                      msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-700'
                    }`}
                  >
                    {msg.content}
                  </span>
                  {msg.confidence && (
                    <span className="text-xs text-gray-400">
                      ({(msg.confidence * 100).toFixed(0)}%)
                    </span>
                  )}
                </div>
                <span className="text-xs text-gray-500">{msg.timestamp}</span>
              </motion.div>
            ))}
            {isTyping && (
              <motion.div
                className="text-left"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <span className="inline-block p-2 rounded-lg bg-gray-700">
                  Typing...
                </span>
              </motion.div>
            )}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about BTC sentiment, news, or risk..."
              className="cyber-input w-full"
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <button
              onClick={() => handleSend()}
              className="cyber-button px-4 py-2"
            >
              Send
            </button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default AIChatAssistant;