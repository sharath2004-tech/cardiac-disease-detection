import { Activity, TrendingUp, TrendingDown, Calendar, FileText, Heart, Brain } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { RiskIndicator } from "@/components/ui/risk-indicator";
import { StatsCard } from "@/components/ui/stats-card";
import { Timeline } from "@/components/ui/timeline";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";

const trendData = [
  { month: "Aug", risk: 28 },
  { month: "Sep", risk: 32 },
  { month: "Oct", risk: 30 },
  { month: "Nov", risk: 35 },
  { month: "Dec", risk: 38 },
  { month: "Jan", risk: 42 },
];

const timelineItems = [
  {
    id: "1",
    title: "Latest Assessment",
    description: "Risk score increased to 42% due to elevated blood pressure",
    date: "Jan 15, 2024",
    icon: Activity,
    iconColor: "text-warning",
  },
  {
    id: "2",
    title: "ECG Report Uploaded",
    description: "Annual ECG showing normal sinus rhythm",
    date: "Jan 10, 2024",
    icon: FileText,
    iconColor: "text-primary",
  },
  {
    id: "3",
    title: "Previous Assessment",
    description: "Risk score was 38% - within moderate range",
    date: "Dec 20, 2023",
    icon: Activity,
    iconColor: "text-muted-foreground",
  },
  {
    id: "4",
    title: "Lab Results Added",
    description: "Cholesterol levels slightly elevated",
    date: "Dec 15, 2023",
    icon: FileText,
    iconColor: "text-primary",
  },
];

export default function Dashboard() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1 container py-8">
        <div className="animate-fade-in">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Prediction Results</h1>
            <p className="text-muted-foreground">
              Your AI-powered cardiac risk assessment from January 15, 2024
            </p>
          </div>

          {/* Main Risk Indicator */}
          <div className="grid lg:grid-cols-3 gap-6 mb-8">
            <div className="lg:col-span-2">
              <RiskIndicator level="medium" score={42} />
            </div>

            <div className="space-y-4">
              <StatsCard
                title="Blood Pressure"
                value="142/92"
                subtitle="mmHg - Elevated"
                icon={Heart}
                variant="warning"
                trend={{ value: 8, isPositive: false }}
              />
              <StatsCard
                title="Heart Rate"
                value="78"
                subtitle="bpm - Normal"
                icon={Activity}
                variant="success"
              />
            </div>
          </div>

          {/* AI Explanation */}
          <Card className="mb-8 border-primary/20 bg-primary/5">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-lg">
                <Brain className="h-5 w-5 text-primary" />
                AI Analysis Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-foreground leading-relaxed">
                <strong>Risk increased due to rising blood pressure</strong> compared to your last assessment. 
                Your systolic BP has increased from 130 to 142 mmHg over the past month. 
                Combined with your cholesterol levels (slightly elevated at 215 mg/dL) and family history, 
                the AI model predicts a moderate cardiac risk. 
                <strong className="text-primary"> Recommendation:</strong> Schedule a follow-up with your 
                cardiologist and consider lifestyle modifications including dietary changes and regular exercise.
              </p>
            </CardContent>
          </Card>

          {/* Stats Grid */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <StatsCard
              title="Cholesterol"
              value="215"
              subtitle="mg/dL"
              icon={Activity}
              variant="warning"
            />
            <StatsCard
              title="Glucose"
              value="102"
              subtitle="mg/dL - Normal"
              icon={Activity}
              variant="success"
            />
            <StatsCard
              title="BMI"
              value="26.4"
              subtitle="Overweight"
              icon={Activity}
              variant="warning"
            />
            <StatsCard
              title="Assessments"
              value="12"
              subtitle="Total completed"
              icon={Calendar}
            />
          </div>

          {/* Charts and Timeline */}
          <div className="grid lg:grid-cols-2 gap-6 mb-8">
            {/* Risk Trend Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  Risk Trend Over Time
                </CardTitle>
                <CardDescription>
                  Historical comparison of your cardiac risk score
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trendData}>
                      <defs>
                        <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="hsl(199, 89%, 48%)" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="hsl(199, 89%, 48%)" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                      <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="risk"
                        stroke="hsl(199, 89%, 48%)"
                        strokeWidth={2}
                        fill="url(#riskGradient)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Assessment Timeline */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="h-5 w-5 text-primary" />
                  Assessment History
                </CardTitle>
                <CardDescription>
                  Timeline of your recent health activities
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Timeline items={timelineItems} />
              </CardContent>
            </Card>
          </div>

          {/* Actions */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="default" size="lg" className="gap-2">
              <FileText className="h-5 w-5" />
              Download Full Report
            </Button>
            <Button variant="outline" size="lg" className="gap-2">
              <Calendar className="h-5 w-5" />
              Schedule Follow-up
            </Button>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
